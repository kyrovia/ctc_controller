#!/usr/bin/env python3
"""Demo 2: UR5e hits an obstacle; live torque plot shows saturation at max_tau."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from demo_common import (
    DEFAULT_MJCF,
    DEFAULT_SCENE_OBSTACLE,
    NV,
    Q_HOME,
    START_KEYCODE,
    TrajectoryRef,
    build_index_maps,
    fk_position,
    ik_position,
    load_mujoco_scene,
    load_pinocchio_model,
    make_obstacle_demo_ctc,
    mujoco_to_pin_position,
    TAU_LIMIT,
    TAU_LIMIT_STR,
    read_named,
    run_realtime_sleep,
    write_named,
)

ROOT = Path(__file__).resolve().parents[1]
JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_OBSTACLE)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--approach-time", type=float, default=2.5)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--history", type=float, default=3.0, help="torque plot window [s]")
    parser.add_argument(
        "--max-tau",
        default=TAU_LIMIT_STR,
        help="six comma-separated UR5e joint torque limits [Nm]",
    )
    return parser.parse_args()


def build_push_target(
    pin_model,
    pin_data,
    frame_id: int,
    obstacle_center: np.ndarray,
    push_beyond: float = 0.15,
) -> np.ndarray:
    home = fk_position(pin_model, pin_data, frame_id, Q_HOME)
    direction = obstacle_center - home
    # Aim beyond the obstacle so contact creates persistent tracking error.
    target = obstacle_center + push_beyond * direction / np.linalg.norm(direction)
    return ik_position(
        pin_model,
        pin_data,
        frame_id,
        target,
        Q_HOME,
        target_rot=None,
    )


def smooth_joint_ref(t: float, q_target: np.ndarray, approach_time: float) -> TrajectoryRef:
    """Quintic minimum-jerk move, then hold the unreachable target."""
    if t >= approach_time:
        return TrajectoryRef(q_target.copy(), np.zeros(NV), np.zeros(NV))

    u = max(0.0, t / approach_time)
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / approach_time
    dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / approach_time**2
    delta = q_target - Q_HOME
    return TrajectoryRef(
        Q_HOME + s * delta,
        ds * delta,
        dds * delta,
    )


class TorquePlot:
    def __init__(self, history_s: float, dt: float, tau_limit: np.ndarray) -> None:
        self.history_s = history_s
        self.max_len = max(10, int(round(history_s / dt)))
        self.times: deque[float] = deque(maxlen=self.max_len)
        self.tau_cmd = [deque(maxlen=self.max_len) for _ in range(NV)]
        self.tau_raw = [deque(maxlen=self.max_len) for _ in range(NV)]
        self.fig, self.axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
        self.fig.canvas.manager.set_window_title("UR5e joint torque")
        self.lines_cmd = []
        self.lines_raw = []
        for i, ax in enumerate(self.axes.flat):
            (line_cmd,) = ax.plot([], [], lw=1.6, color="#1565c0", label="tau (limited)")
            (line_raw,) = ax.plot([], [], lw=1.0, color="#ef6c00", alpha=0.65, label="tau (raw)")
            ax.axhline(tau_limit[i], color="#c62828", ls="--", lw=1.0)
            ax.axhline(-tau_limit[i], color="#c62828", ls="--", lw=1.0)
            ax.text(
                0.02,
                0.92,
                f"max_tau={tau_limit[i]:.0f}",
                transform=ax.transAxes,
                fontsize=8,
                color="#c62828",
            )
            ax.set_ylabel(f"{JOINT_LABELS[i]} [Nm]")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-tau_limit[i] * 1.25, tau_limit[i] * 1.25)
            self.lines_cmd.append(line_cmd)
            self.lines_raw.append(line_raw)
        self.axes[-1, 0].set_xlabel("time [s]")
        self.axes[-1, 1].set_xlabel("time [s]")
        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)

    def update(self, t: float, tau: np.ndarray, tau_raw: np.ndarray) -> None:
        self.times.append(t)
        for i in range(NV):
            self.tau_cmd[i].append(tau[i])
            self.tau_raw[i].append(tau_raw[i])
        t_arr = np.array(self.times)
        for i in range(NV):
            self.lines_cmd[i].set_data(t_arr, np.array(self.tau_cmd[i]))
            self.lines_raw[i].set_data(t_arr, np.array(self.tau_raw[i]))
            self.axes.flat[i].set_xlim(max(0.0, t - self.history_s), max(self.history_s, t))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def main() -> int:
    args = parse_args()
    tau_limit = np.fromstring(args.max_tau, sep=",")
    if tau_limit.shape != (NV,) or np.any(tau_limit <= 0.0):
        print("--max-tau requires six positive comma-separated values", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 1
    if not args.scene.exists():
        print(f"missing scene: {args.scene}", file=sys.stderr)
        print("build first: cmake --build build --target ctc_mujoco_scene_obstacle", file=sys.stderr)
        return 1

    pin_model, pin_data, frame_id = load_pinocchio_model(args.model)
    mj_model, mj_data = load_mujoco_scene(args.scene)
    dt = mj_model.opt.timestep

    qpos_adr, dof_adr, act_adr, _ = build_index_maps(mj_model)
    obstacle_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "obstacle"
    )
    obstacle_geom_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_box"
    )
    if obstacle_body_id < 0 or obstacle_geom_id < 0:
        raise RuntimeError("obstacle scene has no obstacle body/geom")
    obstacle_center = mj_model.body_pos[obstacle_body_id].copy()
    q_target = build_push_target(
        pin_model,
        pin_data,
        frame_id,
        mujoco_to_pin_position(obstacle_center),
    )
    ctc = make_obstacle_demo_ctc(pin_model, pin_data, tau_limit=tau_limit)
    plot = TorquePlot(args.history, dt, tau_limit)

    steps = max(1, int(round(args.duration / dt)))
    sim_started = {"value": False}
    contact_seen = False
    contact_announced = False
    first_contact_time: float | None = None
    sat_count = 0

    def reset_state() -> None:
        write_named(mj_data.qpos, qpos_adr, Q_HOME)
        write_named(mj_data.qvel, dof_adr, np.zeros(NV))
        mujoco.mj_forward(mj_model, mj_data)
        ctc.reset()

    def on_key(keycode: int) -> None:
        if keycode == START_KEYCODE:
            sim_started["value"] = True

    print("UR5e obstacle + torque limit demo")
    print(f"  demo max_tau = {tau_limit} Nm")
    print("  red box = obstacle, blue curve = limited tau, orange = raw tau")
    print("  dashed red lines = max_tau")
    print("  >> 按空格键开始仿真…")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=on_key) as viewer:
        reset_state()
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = obstacle_center

        while viewer.is_running() and not sim_started["value"]:
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
            ref = smooth_joint_ref(sim_t, q_target, args.approach_time)
            q = read_named(mj_data.qpos, qpos_adr)
            dq = read_named(mj_data.qvel, dof_adr)
            result = ctc.step(q, dq, ref, dt)
            if result.saturated:
                sat_count += 1
            for act, val in zip(act_adr, result.tau):
                mj_data.ctrl[act] = val
            mujoco.mj_step(mj_model, mj_data)

            touching_obstacle = any(
                mj_data.contact[j].geom1 == obstacle_geom_id
                or mj_data.contact[j].geom2 == obstacle_geom_id
                for j in range(mj_data.ncon)
            )
            if touching_obstacle:
                contact_seen = True
                if not contact_announced:
                    first_contact_time = sim_t
                    print(f"  >> t={sim_t:.3f}s 接触障碍物，继续顶压并观察 tau")
                    contact_announced = True

            # Drawing six Matplotlib axes at 1 kHz blocks the controller loop.
            if i % 20 == 0:
                plot.update(sim_t, result.tau, result.tau_raw)
            viewer.sync()
            if args.real_time:
                run_realtime_sleep(t0, sim_t)

        print(f"  contact detected: {contact_seen}")
        if first_contact_time is not None:
            print(f"  first obstacle contact: {first_contact_time:.3f}s")
        print(f"  saturated steps: {sat_count}/{steps}")
        print("  >> 演示结束，请关闭窗口退出")
        while viewer.is_running():
            viewer.sync()
            plt.pause(0.02)

    plt.ioff()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
