#!/usr/bin/env python3
"""UR5e MuJoCo viewer: visualize computed torque control (CTC) on Menagerie UR5e."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = Path(
    "/home/l/gravity_comp/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
DEFAULT_SCENE = ROOT / "build" / "models" / "ur5e_menagerie_torque.mjb"

UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

Q_HOME = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
NV = len(UR5E_JOINTS)
TAU_LIMIT = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
START_KEYCODE = 32  # space

MODE_HELP = {
    "no-ctc": "无控制 (tau=0)，机械臂会因重力下塌",
    "hold": "CTC 保持在 q_home",
    "track": "CTC 跟踪 q_home 附近的正弦轨迹",
    "compare": "依次 no-ctc → hold → track",
}


@dataclass
class Gains:
    kp: np.ndarray
    kd: np.ndarray
    ki: np.ndarray = field(default_factory=lambda: np.zeros(NV))


@dataclass
class Safety:
    tau_abs_max: np.ndarray = field(default_factory=lambda: TAU_LIMIT.copy())
    max_delta_tau: float = 40.0
    hold_on_nonfinite: bool = True
    integral_limit: float = 2.0


@dataclass
class TrajectoryRef:
    qd: np.ndarray
    dqd: np.ndarray
    ddqd: np.ndarray


class CTCController:
    """Python mirror of ctc_control::CTCController."""

    def __init__(
        self,
        pin_model: pin.Model,
        pin_data: pin.Data,
        gains: Gains,
        safety: Safety | None = None,
    ) -> None:
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.gains = gains
        self.safety = safety or Safety()
        self.e_int = np.zeros(NV)
        self.tau_prev = np.zeros(NV)
        self.has_prev = False

    def reset(self) -> None:
        self.e_int = np.zeros(NV)
        self.tau_prev = np.zeros(NV)
        self.has_prev = False

    @staticmethod
    def _saturate_abs(tau: np.ndarray, limit: np.ndarray) -> np.ndarray:
        return np.clip(tau, -limit, limit)

    @staticmethod
    def _saturate_rate(tau: np.ndarray, prev: np.ndarray, max_delta: float) -> np.ndarray:
        delta = np.clip(tau - prev, -max_delta, max_delta)
        return prev + delta

    def step(self, q: np.ndarray, dq: np.ndarray, ref: TrajectoryRef, dt: float) -> np.ndarray:
        if not (q.shape == (NV,) and dq.shape == (NV,) and ref.qd.shape == (NV,)):
            return self.tau_prev.copy()

        err = ref.qd - q
        derr = ref.dqd - dq

        if self.gains.ki.size == NV and dt > 0.0:
            self.e_int += err * dt
            if self.safety.integral_limit > 0.0:
                lim = np.full(NV, self.safety.integral_limit)
                self.e_int = self._saturate_abs(self.e_int, lim)

        qdd_cmd = ref.ddqd + self.gains.kp * err + self.gains.kd * derr
        if self.gains.ki.size == NV:
            qdd_cmd += self.gains.ki * self.e_int

        tau = pin.rnea(self.pin_model, self.pin_data, q, dq, qdd_cmd)

        if self.safety.hold_on_nonfinite and not np.all(np.isfinite(tau)):
            return self.tau_prev.copy()
        if self.safety.tau_abs_max.size == NV:
            tau = self._saturate_abs(tau, self.safety.tau_abs_max)
        if self.has_prev and self.safety.max_delta_tau > 0.0:
            tau = self._saturate_rate(tau, self.tau_prev, self.safety.max_delta_tau)

        self.tau_prev = tau.copy()
        self.has_prev = True
        return tau


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {name:8} {text}" for name, text in MODE_HELP.items()),
    )
    parser.add_argument("--mode", choices=list(MODE_HELP), default="track")
    parser.add_argument("--model", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--headless", action="store_true", help="run without viewer; for automated test")
    parser.add_argument("--max-mean-err", type=float, default=0.35, help="headless pass threshold [rad]")
    return parser.parse_args(argv)


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "no-ctc":
        args.duration = 3.0 if args.duration is None else args.duration
    elif args.mode == "hold":
        args.duration = 5.0 if args.duration is None else args.duration
    elif args.mode == "track":
        args.duration = 5.0 if args.duration is None else args.duration
    elif args.mode == "compare":
        args.real_time = True
        args.duration = 9.0 if args.duration is None else args.duration


def load_pinocchio_model(model_path: Path) -> tuple[pin.Model, pin.Data]:
    if model_path.suffix.lower() in {".xml", ".mjcf"}:
        model = pin.buildModelFromMJCF(str(model_path))
    else:
        model = pin.buildModelFromUrdf(str(model_path))
    return model, model.createData()


def joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"MuJoCo model has no joint named '{name}'")
    return jid


def actuator_for_joint(model: mujoco.MjModel, joint_id: int) -> int:
    for act in range(model.nu):
        if (
            int(model.actuator_trntype[act]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[act, 0]) == joint_id
        ):
            return act
    raise RuntimeError(f"MuJoCo model has no motor for joint id {joint_id}")


def build_index_maps(model: mujoco.MjModel):
    qpos_adr = [model.jnt_qposadr[joint_id(model, name)] for name in UR5E_JOINTS]
    dof_adr = [model.jnt_dofadr[joint_id(model, name)] for name in UR5E_JOINTS]
    act_adr = [actuator_for_joint(model, joint_id(model, name)) for name in UR5E_JOINTS]
    return qpos_adr, dof_adr, act_adr


def read_named(src: np.ndarray, adr: list[int]) -> np.ndarray:
    return np.array([src[a] for a in adr], dtype=float)


def write_named(dst: np.ndarray, adr: list[int], values: np.ndarray) -> None:
    for a, val in zip(adr, values):
        dst[a] = val


def sinusoid_ref(t: float, center: np.ndarray) -> TrajectoryRef:
    qd = center.copy()
    dqd = np.zeros(NV)
    ddqd = np.zeros(NV)
    for i in range(NV):
        w = 2.0 * math.pi * (1.0 + 0.15 * i)
        amp = 0.15 / (i + 1)
        phase = 0.5 * i
        qd[i] = center[i] + amp * math.sin(w * t + phase)
        dqd[i] = amp * w * math.cos(w * t + phase)
        ddqd[i] = -amp * w * w * math.sin(w * t + phase)
    return TrajectoryRef(qd=qd, dqd=dqd, ddqd=ddqd)


def home_ref() -> TrajectoryRef:
    zero = np.zeros(NV)
    return TrajectoryRef(qd=Q_HOME.copy(), dqd=zero.copy(), ddqd=zero.copy())


def make_ctc(pin_model: pin.Model, pin_data: pin.Data) -> CTCController:
    gains = Gains(kp=np.full(NV, 120.0), kd=np.full(NV, 24.0))
    safety = Safety()
    return CTCController(pin_model, pin_data, gains, safety)


def make_compare_controller(
    ctc: CTCController,
    reset_state,
) -> tuple[callable, callable]:
    phase1_end = 3.0
    phase2_end = 6.0
    announced = {"p1": False, "p2": False, "p3": False}

    def ref_fn(t: float) -> TrajectoryRef:
        if t < phase1_end:
            return home_ref()
        if t < phase2_end:
            return home_ref()
        return sinusoid_ref(t, Q_HOME)

    def tau_fn(t: float, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if t < phase1_end:
            if not announced["p1"]:
                print("  >> [no-ctc] tau = 0")
                announced["p1"] = True
            return np.zeros(NV)
        if t < phase2_end:
            if not announced["p2"]:
                reset_state()
                ctc.reset()
                print("  >> [hold] CTC at q_home")
                announced["p2"] = True
            return ctc.step(q, dq, home_ref(), 0.001)
        if not announced["p3"]:
            print("  >> [track] CTC sinusoid tracking")
            announced["p3"] = True
        return ctc.step(q, dq, sinusoid_ref(t, Q_HOME), 0.001)

    return ref_fn, tau_fn


def make_single_controller(mode: str, ctc: CTCController):
    def ref_fn(t: float) -> TrajectoryRef:
        if mode == "track":
            return sinusoid_ref(t, Q_HOME)
        return home_ref()

    def tau_fn(t: float, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if mode == "no-ctc":
            return np.zeros(NV)
        return ctc.step(q, dq, ref_fn(t), 0.001)

    return ref_fn, tau_fn


def evaluate_result(mode: str, max_err: float, mean_err: float, max_mean_err: float) -> bool:
    print(f"mean |e| = {mean_err:.6f} rad, max |e| = {max_err:.6f} rad")
    if mode == "no-ctc":
        print("expect large drift without control")
        return max_err > 0.05
    ok = mean_err < max_mean_err
    if not ok:
        print(f"FAIL: mean |e| >= {max_mean_err:.3f} rad")
    else:
        print("PASS")
    return ok


def run_simulation(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_adr: list[int],
    dof_adr: list[int],
    act_adr: list[int],
    ref_fn,
    tau_fn,
    duration: float,
    real_time: bool,
) -> tuple[float, float]:
    steps = max(1, int(round(duration / mj_model.opt.timestep)))
    err_sum = 0.0
    max_err = 0.0

    write_named(mj_data.qpos, qpos_adr, Q_HOME)
    write_named(mj_data.qvel, dof_adr, np.zeros(NV))
    mujoco.mj_forward(mj_model, mj_data)

    t0 = time.time()
    for i in range(steps):
        t = i * mj_model.opt.timestep
        q = read_named(mj_data.qpos, qpos_adr)
        dq = read_named(mj_data.qvel, dof_adr)
        ref = ref_fn(t)
        err = float(np.linalg.norm(ref.qd - q))
        err_sum += err
        max_err = max(max_err, err)

        tau = tau_fn(t, q, dq)
        for act, val in zip(act_adr, tau):
            mj_data.ctrl[act] = val
        mujoco.mj_step(mj_model, mj_data)

        if real_time:
            target = t0 + t
            while time.time() < target:
                time.sleep(0.0005)

    return max_err, err_sum / steps


def run_with_viewer(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_adr: list[int],
    dof_adr: list[int],
    act_adr: list[int],
    ref_fn,
    tau_fn,
    args: argparse.Namespace,
) -> tuple[float, float]:
    steps = max(1, int(round(args.duration / mj_model.opt.timestep)))
    err_sum = 0.0
    max_err = 0.0
    sim_started = {"value": False}

    def reset_state() -> None:
        write_named(mj_data.qpos, qpos_adr, Q_HOME)
        write_named(mj_data.qvel, dof_adr, np.zeros(NV))
        mujoco.mj_forward(mj_model, mj_data)

    def on_key(keycode: int) -> None:
        if keycode == START_KEYCODE:
            sim_started["value"] = True

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=on_key) as viewer:
        reset_state()
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        print("  >> 按空格键开始仿真…")
        while viewer.is_running() and not sim_started["value"]:
            viewer.sync()
            time.sleep(0.01)
        if not viewer.is_running():
            return 0.0, 0.0

        print("  >> 仿真开始")
        t0 = time.time()
        i = 0
        sim_finished = False

        while viewer.is_running():
            if i < steps:
                t = i * mj_model.opt.timestep
                q = read_named(mj_data.qpos, qpos_adr)
                dq = read_named(mj_data.qvel, dof_adr)
                ref = ref_fn(t)
                err = float(np.linalg.norm(ref.qd - q))
                err_sum += err
                max_err = max(max_err, err)

                tau = tau_fn(t, q, dq)
                for act, val in zip(act_adr, tau):
                    mj_data.ctrl[act] = val
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                if args.real_time:
                    target = t0 + t
                    while time.time() < target and viewer.is_running():
                        time.sleep(0.0005)
                i += 1
            else:
                if not sim_finished:
                    print("  >> 演示结束，请关闭窗口退出")
                    sim_finished = True
                viewer.sync()
                time.sleep(0.01)

    return max_err, err_sum / steps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    apply_mode_defaults(args)

    if not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 1
    if not args.scene.exists():
        print(f"missing scene: {args.scene}", file=sys.stderr)
        print("build first: cmake -S . -B build && cmake --build build", file=sys.stderr)
        return 1

    pin_model, pin_data = load_pinocchio_model(args.model)
    pin_names = [
        pin_model.names[i]
        for i in range(1, pin_model.njoints)
        if pin_model.joints[i].nq > 0
    ]
    if pin_names != UR5E_JOINTS:
        print(f"Pinocchio joints {pin_names} != expected {UR5E_JOINTS}", file=sys.stderr)
        return 1

    if args.scene.suffix == ".mjb":
        mj_model = mujoco.MjModel.from_binary_path(str(args.scene))
    else:
        mj_model = mujoco.MjModel.from_xml_path(str(args.scene))
    mj_data = mujoco.MjData(mj_model)
    qpos_adr, dof_adr, act_adr = build_index_maps(mj_model)

    ctc = make_ctc(pin_model, pin_data)

    def reset_state() -> None:
        write_named(mj_data.qpos, qpos_adr, Q_HOME)
        write_named(mj_data.qvel, dof_adr, np.zeros(NV))
        mujoco.mj_forward(mj_model, mj_data)

    if args.mode == "compare":
        ref_fn, tau_fn = make_compare_controller(ctc, reset_state)
    else:
        ctc.reset()
        ref_fn, tau_fn = make_single_controller(args.mode, ctc)

    print("UR5e CTC MuJoCo viewer (Menagerie)")
    print(f"  Model: {args.model}")
    print(f"  Scene: {args.scene}")
    print(f"  Mode : {args.mode} — {MODE_HELP[args.mode]}")
    print(f"  dt={mj_model.opt.timestep}, duration={args.duration}s")

    if args.headless:
        max_err, mean_err = run_simulation(
            mj_model, mj_data, qpos_adr, dof_adr, act_adr, ref_fn, tau_fn,
            args.duration, real_time=False,
        )
    else:
        print("  窗口打开后按空格开始；演示结束后请手动关闭窗口退出。")
        max_err, mean_err = run_with_viewer(
            mj_model, mj_data, qpos_adr, dof_adr, act_adr, ref_fn, tau_fn, args,
        )

    ok = evaluate_result(args.mode, max_err, mean_err, args.max_mean_err)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
