#!/usr/bin/env python3
"""Demo 1: benchmark UR5e CTC on a high-speed Cartesian circle."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from demo_common import (
    DEFAULT_MJCF,
    DEFAULT_SCENE,
    NV,
    Q_HOME,
    START_KEYCODE,
    TAU_LIMIT,
    StepResult,
    add_sphere_geom,
    build_index_maps,
    draw_point_cloud,
    ee_position,
    fk_position,
    ik_position,
    interp_ref,
    load_mujoco_scene,
    load_pinocchio_model,
    make_default_ctc,
    pin_to_mujoco_position,
    read_named,
    run_realtime_sleep,
    write_named,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--period", type=float, help="override circle period [s]")
    parser.add_argument(
        "--tcp-speed", type=float, default=0.6, help="target average TCP speed [m/s]"
    )
    parser.add_argument("--radius", type=float, default=0.08, help="circle radius [m]")
    parser.add_argument(
        "--ik-dt", type=float, default=0.005, help="IK sampling step [s]"
    )
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument(
        "--headless", action="store_true", help="only print benchmark results"
    )
    args = parser.parse_args()
    if args.radius <= 0.0:
        parser.error("--radius must be positive")
    if args.tcp_speed <= 0.0:
        parser.error("--tcp-speed must be positive")
    if args.period is not None and args.period <= 0.0:
        parser.error("--period must be positive")
    if args.period is None:
        args.period = 2.0 * math.pi * args.radius / args.tcp_speed
    return args


def circle_offset(phase: float, radius: float) -> np.ndarray:
    """Circle in the world X-Z plane, starting at the home pose."""
    return np.array(
        [radius * (math.cos(phase) - 1.0), 0.0, radius * math.sin(phase)]
    )


def build_circle_trajectory(
    pin_model,
    pin_data,
    frame_id: int,
    center: np.ndarray,
    radius: float,
    period: float,
    dt: float,
):
    samples = max(3, int(round(period / dt)) + 1)
    times = np.linspace(0.0, period, samples)
    q_samples: list[np.ndarray] = []
    ee_samples: list[np.ndarray] = []
    q_seed = Q_HOME.copy()

    for t in times:
        phase = 2.0 * math.pi * t / period
        target = center + circle_offset(phase, radius)
        q_seed = ik_position(
            pin_model,
            pin_data,
            frame_id,
            target,
            q_seed,
            target_rot=None,
        )
        q_samples.append(q_seed.copy())
        ee_samples.append(fk_position(pin_model, pin_data, frame_id, q_seed))

    q_array = np.unwrap(np.asarray(q_samples), axis=0)
    dq_array = np.gradient(q_array, times, axis=0, edge_order=2)
    ddq_array = np.gradient(dq_array, times, axis=0, edge_order=2)
    q_samples = [row.copy() for row in q_array]
    dq_samples = [row.copy() for row in dq_array]
    ddq_samples = [row.copy() for row in ddq_array]

    return times, q_samples, dq_samples, ddq_samples, ee_samples


@dataclass
class Metrics:
    joint_rmse: float
    joint_max: float
    ee_rmse_mm: float
    ee_max_mm: float
    saturation_percent: float


class PositionPID:
    """Pure joint position PID that outputs torque without model compensation."""

    def __init__(self) -> None:
        self.kp = np.array([180.0, 180.0, 140.0, 35.0, 35.0, 25.0])
        self.kd = np.array([30.0, 30.0, 24.0, 8.0, 8.0, 6.0])
        self.ki = np.array([20.0, 20.0, 15.0, 4.0, 4.0, 3.0])
        self.integral_limit = 1.0
        self.e_int = np.zeros(NV)
        self.tau_prev = np.zeros(NV)
        self.has_prev = False

    def reset(self) -> None:
        self.e_int.fill(0.0)
        self.tau_prev.fill(0.0)
        self.has_prev = False

    def step(self, q: np.ndarray, dq: np.ndarray, ref, dt: float) -> StepResult:
        error = ref.qd - q
        velocity_error = ref.dqd - dq
        self.e_int = np.clip(
            self.e_int + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        tau_raw = self.kp * error + self.kd * velocity_error + self.ki * self.e_int
        tau = np.clip(tau_raw, -TAU_LIMIT, TAU_LIMIT)
        if self.has_prev:
            tau = self.tau_prev + np.clip(tau - self.tau_prev, -40.0, 40.0)
        saturated = not np.allclose(tau, tau_raw)
        self.tau_prev = tau.copy()
        self.has_prev = True
        return StepResult(tau=tau, tau_raw=tau_raw, saturated=saturated)


def run_trial(
    mj_model,
    pin_model,
    pin_data,
    qpos_adr: list[int],
    dof_adr: list[int],
    act_adr: list[int],
    site_id: int,
    frame_id: int,
    controller,
    times: np.ndarray,
    q_samples: list[np.ndarray],
    dq_samples: list[np.ndarray],
    ddq_samples: list[np.ndarray],
    period: float,
    duration: float,
    dt: float,
) -> Metrics:
    """Run one deterministic trial and return tracking-error metrics."""
    data = mujoco.MjData(mj_model)
    write_named(data.qpos, qpos_adr, Q_HOME)
    write_named(data.qvel, dof_adr, np.zeros(NV))
    mujoco.mj_forward(mj_model, data)
    controller.reset()

    joint_squared_errors: list[float] = []
    joint_errors: list[float] = []
    ee_squared_errors: list[float] = []
    ee_errors: list[float] = []
    saturated_steps = 0
    steps = max(1, int(round(duration / dt)))

    for i in range(steps):
        loop_t = (i * dt) % period
        ref = interp_ref(times, q_samples, dq_samples, ddq_samples, loop_t)

        q = read_named(data.qpos, qpos_adr)
        dq = read_named(data.qvel, dof_adr)
        result = controller.step(q, dq, ref, dt)
        for act, value in zip(act_adr, result.tau):
            data.ctrl[act] = value
        mujoco.mj_step(mj_model, data)

        joint_error = float(np.linalg.norm(ref.qd - q))
        ee_ref = pin_to_mujoco_position(
            fk_position(
                pin_model,
                pin_data,
                frame_id,
                ref.qd,
            )
        )
        ee_error = float(np.linalg.norm(ee_position(data, site_id) - ee_ref))
        joint_squared_errors.append(joint_error**2)
        joint_errors.append(joint_error)
        ee_squared_errors.append(ee_error**2)
        ee_errors.append(ee_error)
        saturated_steps += int(result.saturated)

    return Metrics(
        joint_rmse=math.sqrt(float(np.mean(joint_squared_errors))),
        joint_max=max(joint_errors),
        ee_rmse_mm=1000.0 * math.sqrt(float(np.mean(ee_squared_errors))),
        ee_max_mm=1000.0 * max(ee_errors),
        saturation_percent=100.0 * saturated_steps / steps,
    )


def print_comparison(ctc_metrics: Metrics, pid_metrics: Metrics) -> None:
    """Print compact quantitative tracking results."""
    improvement = 100.0 * (
        1.0 - ctc_metrics.ee_rmse_mm / max(pid_metrics.ee_rmse_mm, 1e-12)
    )
    print("\n高速跟踪定量结果")
    print("  控制器                 关节RMSE   关节最大误差   末端RMSE   末端最大误差   饱和")
    print(
        f"  CTC（完整前馈）        {ctc_metrics.joint_rmse:8.4f} rad"
        f"  {ctc_metrics.joint_max:10.4f} rad"
        f"  {ctc_metrics.ee_rmse_mm:8.2f} mm"
        f"  {ctc_metrics.ee_max_mm:10.2f} mm"
        f"  {ctc_metrics.saturation_percent:5.1f}%"
    )
    print(
        f"  纯位置 PID             {pid_metrics.joint_rmse:8.4f} rad"
        f"  {pid_metrics.joint_max:10.4f} rad"
        f"  {pid_metrics.ee_rmse_mm:8.2f} mm"
        f"  {pid_metrics.ee_max_mm:10.2f} mm"
        f"  {pid_metrics.saturation_percent:5.1f}%"
    )
    print(f"  CTC 末端 RMSE 改善：{improvement:.1f}%\n")


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 1
    if not args.scene.exists():
        print(f"missing scene: {args.scene}", file=sys.stderr)
        print("build first: cmake --build build --target ctc_mujoco_scene", file=sys.stderr)
        return 1

    pin_model, pin_data, frame_id = load_pinocchio_model(args.model)
    center = fk_position(pin_model, pin_data, frame_id, Q_HOME)
    dt = 0.001

    times, q_samples, dq_samples, ddq_samples, ee_ref_points_pin = build_circle_trajectory(
        pin_model,
        pin_data,
        frame_id,
        center,
        args.radius,
        args.period,
        args.ik_dt,
    )
    ee_ref_points = [pin_to_mujoco_position(point) for point in ee_ref_points_pin]
    viewer_center = pin_to_mujoco_position(center)

    mj_model, mj_data = load_mujoco_scene(args.scene)
    qpos_adr, dof_adr, act_adr, site_id = build_index_maps(mj_model)
    ctc = make_default_ctc(pin_model, pin_data)
    pid = PositionPID()

    print(
        f"高速圆轨迹：半径 {args.radius:.3f} m，周期 {args.period:.2f} s，"
        f"末端平均速度约 {2.0 * math.pi * args.radius / args.period:.2f} m/s"
    )
    ctc_metrics = run_trial(
        mj_model,
        pin_model,
        pin_model.createData(),
        qpos_adr,
        dof_adr,
        act_adr,
        site_id,
        frame_id,
        ctc,
        times,
        q_samples,
        dq_samples,
        ddq_samples,
        args.period,
        args.duration,
        dt,
    )
    pid_metrics = run_trial(
        mj_model,
        pin_model,
        pin_model.createData(),
        qpos_adr,
        dof_adr,
        act_adr,
        site_id,
        frame_id,
        pid,
        times,
        q_samples,
        dq_samples,
        ddq_samples,
        args.period,
        args.duration,
        dt,
    )
    print_comparison(ctc_metrics, pid_metrics)
    if args.headless:
        return 0

    steps = max(1, int(round(args.duration / dt)))
    sim_started = {"value": False}

    def reset_state() -> None:
        write_named(mj_data.qpos, qpos_adr, Q_HOME)
        write_named(mj_data.qvel, dof_adr, np.zeros(NV))
        mujoco.mj_forward(mj_model, mj_data)
        ctc.reset()

    def on_key(keycode: int) -> None:
        if keycode == START_KEYCODE:
            sim_started["value"] = True

    print("UR5e 高速 Cartesian circle CTC demo")
    print(f"  start={viewer_center}, radius={args.radius} m, period={args.period} s")
    print("  green dots = reference circle, orange = actual EE trail")
    print("  >> 按空格键开始仿真…")

    ee_trail: list[np.ndarray] = []

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=on_key) as viewer:
        reset_state()
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = viewer_center

        while viewer.is_running() and not sim_started["value"]:
            draw_point_cloud(viewer.user_scn, ee_ref_points, 0.008, [0.1, 0.85, 0.25, 0.85])
            viewer.sync()
            time.sleep(0.01)
        if not viewer.is_running():
            return 0

        print("  >> 仿真开始")
        t0 = time.time()
        for i in range(steps):
            if not viewer.is_running():
                break
            sim_t = i * dt
            loop_t = sim_t % args.period
            ref = interp_ref(times, q_samples, dq_samples, ddq_samples, loop_t)

            q = read_named(mj_data.qpos, qpos_adr)
            dq = read_named(mj_data.qvel, dof_adr)
            result = ctc.step(q, dq, ref, dt)
            for act, val in zip(act_adr, result.tau):
                mj_data.ctrl[act] = val
            mujoco.mj_step(mj_model, mj_data)

            ee_now = ee_position(mj_data, site_id)
            ee_trail.append(ee_now)
            if len(ee_trail) > 400:
                ee_trail.pop(0)

            draw_point_cloud(viewer.user_scn, ee_ref_points, 0.008, [0.1, 0.85, 0.25, 0.85])
            draw_point_cloud(viewer.user_scn, ee_trail, 0.007, [1.0, 0.55, 0.05, 0.9], start_idx=len(ee_ref_points))
            add_idx = len(ee_ref_points) + len(ee_trail)
            if add_idx < viewer.user_scn.maxgeom:
                add_sphere_geom(viewer.user_scn, add_idx, ee_now, 0.012, [0.95, 0.95, 0.1, 1.0])
                viewer.user_scn.ngeom = add_idx + 1
            viewer.sync()

            if args.real_time:
                run_realtime_sleep(t0, sim_t)

        print("  >> 演示结束，请关闭窗口退出")
        while viewer.is_running():
            draw_point_cloud(viewer.user_scn, ee_ref_points, 0.008, [0.1, 0.85, 0.25, 0.85])
            draw_point_cloud(viewer.user_scn, ee_trail, 0.007, [1.0, 0.55, 0.05, 0.9], start_idx=len(ee_ref_points))
            viewer.sync()
            time.sleep(0.02)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
