#!/usr/bin/env python3
"""Demo 1: UR5e CTC tracks a Cartesian circle shown in MuJoCo."""

from __future__ import annotations

import argparse
import math
import sys
import time
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
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--period", type=float, default=8.0, help="seconds for one circle")
    parser.add_argument("--radius", type=float, default=0.08, help="circle radius [m]")
    parser.add_argument("--ik-dt", type=float, default=0.01, help="IK sampling step for precompute [s]")
    parser.add_argument("--real-time", action="store_true")
    return parser.parse_args()


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

    print("UR5e Cartesian circle CTC demo")
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
        err_sum = 0.0
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

            err_sum += float(np.linalg.norm(ref.qd - q))
            draw_point_cloud(viewer.user_scn, ee_ref_points, 0.008, [0.1, 0.85, 0.25, 0.85])
            draw_point_cloud(viewer.user_scn, ee_trail, 0.007, [1.0, 0.55, 0.05, 0.9], start_idx=len(ee_ref_points))
            add_idx = len(ee_ref_points) + len(ee_trail)
            if add_idx < viewer.user_scn.maxgeom:
                add_sphere_geom(viewer.user_scn, add_idx, ee_now, 0.012, [0.95, 0.95, 0.1, 1.0])
                viewer.user_scn.ngeom = add_idx + 1
            viewer.sync()

            if args.real_time:
                run_realtime_sleep(t0, sim_t)

        mean_err = err_sum / max(steps, 1)
        print(f"  mean joint |e| = {mean_err:.4f} rad")
        print("  >> 演示结束，请关闭窗口退出")
        while viewer.is_running():
            draw_point_cloud(viewer.user_scn, ee_ref_points, 0.008, [0.1, 0.85, 0.25, 0.85])
            draw_point_cloud(viewer.user_scn, ee_trail, 0.007, [1.0, 0.55, 0.05, 0.9], start_idx=len(ee_ref_points))
            viewer.sync()
            time.sleep(0.02)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
