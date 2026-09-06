#!/usr/bin/env python3
"""Shared helpers for UR5e CTC MuJoCo demos."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = Path(
    "/home/l/gravity_comp/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
DEFAULT_SCENE = ROOT / "build" / "models" / "ur5e_menagerie_torque.mjb"
DEFAULT_SCENE_OBSTACLE = ROOT / "build" / "models" / "ur5e_menagerie_torque_obstacle.mjb"

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
EE_FRAME = "attachment_site"
EE_SITE = "attachment_site"
START_KEYCODE = 32


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


@dataclass
class StepResult:
    tau: np.ndarray
    tau_raw: np.ndarray
    saturated: bool


class CTCController:
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
        self.tau_raw_prev = np.zeros(NV)
        self.has_prev = False

    def reset(self) -> None:
        self.e_int = np.zeros(NV)
        self.tau_prev = np.zeros(NV)
        self.tau_raw_prev = np.zeros(NV)
        self.has_prev = False

    @staticmethod
    def _saturate_abs(tau: np.ndarray, limit: np.ndarray) -> np.ndarray:
        return np.clip(tau, -limit, limit)

    @staticmethod
    def _saturate_rate(tau: np.ndarray, prev: np.ndarray, max_delta: float) -> np.ndarray:
        delta = np.clip(tau - prev, -max_delta, max_delta)
        return prev + delta

    def step(self, q: np.ndarray, dq: np.ndarray, ref: TrajectoryRef, dt: float) -> StepResult:
        if not (q.shape == (NV,) and dq.shape == (NV,) and ref.qd.shape == (NV,)):
            return StepResult(self.tau_prev.copy(), self.tau_raw_prev.copy(), False)

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

        tau_raw = pin.rnea(self.pin_model, self.pin_data, q, dq, qdd_cmd)
        if self.safety.hold_on_nonfinite and not np.all(np.isfinite(tau_raw)):
            return StepResult(self.tau_prev.copy(), self.tau_raw_prev.copy(), False)

        tau = tau_raw.copy()
        saturated = False
        if self.safety.tau_abs_max.size == NV:
            clipped = self._saturate_abs(tau, self.safety.tau_abs_max)
            saturated = not np.allclose(clipped, tau)
            tau = clipped
        if self.has_prev and self.safety.max_delta_tau > 0.0:
            rated = self._saturate_rate(tau, self.tau_prev, self.safety.max_delta_tau)
            saturated = saturated or not np.allclose(rated, tau)
            tau = rated

        self.tau_raw_prev = tau_raw.copy()
        self.tau_prev = tau.copy()
        self.has_prev = True
        return StepResult(tau=tau, tau_raw=tau_raw, saturated=saturated)


def load_pinocchio_model(model_path: Path) -> tuple[pin.Model, pin.Data, int]:
    if model_path.suffix.lower() in {".xml", ".mjcf"}:
        model = pin.buildModelFromMJCF(str(model_path))
    else:
        model = pin.buildModelFromUrdf(str(model_path))
    data = model.createData()
    frame_id = model.getFrameId(EE_FRAME)
    return model, data, frame_id


def joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"MuJoCo model has no joint named '{name}'")
    return jid


def actuator_for_joint(model: mujoco.MjModel, jid: int) -> int:
    for act in range(model.nu):
        if (
            int(model.actuator_trntype[act]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[act, 0]) == jid
        ):
            return act
    raise RuntimeError(f"MuJoCo model has no motor for joint id {jid}")


def build_index_maps(model: mujoco.MjModel):
    qpos_adr = [model.jnt_qposadr[joint_id(model, name)] for name in UR5E_JOINTS]
    dof_adr = [model.jnt_dofadr[joint_id(model, name)] for name in UR5E_JOINTS]
    act_adr = [actuator_for_joint(model, joint_id(model, name)) for name in UR5E_JOINTS]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if site_id < 0:
        raise RuntimeError(f"MuJoCo model has no site named '{EE_SITE}'")
    return qpos_adr, dof_adr, act_adr, site_id


def read_named(src: np.ndarray, adr: list[int]) -> np.ndarray:
    return np.array([src[a] for a in adr], dtype=float)


def write_named(dst: np.ndarray, adr: list[int], values: np.ndarray) -> None:
    for a, val in zip(adr, values):
        dst[a] = val


def ee_position(mj_data: mujoco.MjData, site_id: int) -> np.ndarray:
    return mj_data.site_xpos[site_id].copy()


def pin_to_mujoco_position(position: np.ndarray) -> np.ndarray:
    """Convert Pinocchio MJCF-parser world coordinates to MuJoCo world coordinates."""
    return np.array([-position[0], -position[1], position[2]])


def mujoco_to_pin_position(position: np.ndarray) -> np.ndarray:
    """Inverse of pin_to_mujoco_position."""
    return np.array([-position[0], -position[1], position[2]])


def make_default_ctc(
    pin_model: pin.Model,
    pin_data: pin.Data,
    tau_limit: np.ndarray | None = None,
) -> CTCController:
    gains = Gains(kp=np.full(NV, 120.0), kd=np.full(NV, 24.0))
    safety = Safety(
        tau_abs_max=TAU_LIMIT.copy() if tau_limit is None else tau_limit.copy()
    )
    return CTCController(pin_model, pin_data, gains, safety)


def home_ref() -> TrajectoryRef:
    zero = np.zeros(NV)
    return TrajectoryRef(qd=Q_HOME.copy(), dqd=zero.copy(), ddqd=zero.copy())


def fk_position(pin_model: pin.Model, pin_data: pin.Data, frame_id: int, q: np.ndarray) -> np.ndarray:
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[frame_id].translation.copy()


def fk_rotation(pin_model: pin.Model, pin_data: pin.Data, frame_id: int, q: np.ndarray) -> np.ndarray:
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[frame_id].rotation.copy()


def ik_position(
    pin_model: pin.Model,
    pin_data: pin.Data,
    frame_id: int,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    target_rot: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    damp: float = 0.05,
    step_scale: float = 0.4,
) -> np.ndarray:
    q = q_init.copy()
    for _ in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        current = pin_data.oMf[frame_id]
        if target_rot is None:
            err = target_pos - current.translation
            if np.linalg.norm(err) < tol:
                break
            jacobian = pin.computeFrameJacobian(
                pin_model, pin_data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damp**2 * np.eye(3), err
            )
        else:
            target_se3 = pin.SE3(target_rot, target_pos)
            err6 = pin.log6(current.inverse() * target_se3).vector
            if np.linalg.norm(err6) < tol:
                break
            jacobian = pin.computeFrameJacobian(
                pin_model, pin_data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damp**2 * np.eye(6), err6
            )
        q = pin.integrate(pin_model, q, dq * step_scale)
    return q


def differentiate_series(values: list[np.ndarray], dt: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = len(values)
    velocities = [np.zeros_like(values[0]) for _ in range(n)]
    accelerations = [np.zeros_like(values[0]) for _ in range(n)]
    for i in range(1, n):
        velocities[i] = (values[i] - values[i - 1]) / dt
    for i in range(1, n - 1):
        accelerations[i] = (velocities[i + 1] - velocities[i - 1]) / (2.0 * dt)
    accelerations[0] = accelerations[1]
    accelerations[-1] = accelerations[-2]
    return velocities, accelerations


def interp_ref(samples_t: np.ndarray, q: list[np.ndarray], dq: list[np.ndarray], ddq: list[np.ndarray], t: float):
    if t <= samples_t[0]:
        idx = 0
        alpha = 0.0
    elif t >= samples_t[-1]:
        idx = len(samples_t) - 2
        alpha = 1.0
    else:
        idx = int(np.searchsorted(samples_t, t, side="right") - 1)
        idx = min(idx, len(samples_t) - 2)
        span = samples_t[idx + 1] - samples_t[idx]
        alpha = 0.0 if span <= 0.0 else (t - samples_t[idx]) / span
    qd = (1.0 - alpha) * q[idx] + alpha * q[idx + 1]
    dqd = (1.0 - alpha) * dq[idx] + alpha * dq[idx + 1]
    ddqd = (1.0 - alpha) * ddq[idx] + alpha * ddq[idx + 1]
    return TrajectoryRef(qd=qd, dqd=dqd, ddqd=ddqd)


def add_sphere_geom(scn: mujoco.MjvScene, idx: int, pos: np.ndarray, radius: float, rgba) -> int:
    if idx >= scn.maxgeom:
        return idx
    geom = scn.geoms[idx]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=pos,
        mat=np.eye(3).reshape(-1),
        rgba=np.array(rgba, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    return idx + 1


def draw_point_cloud(scn: mujoco.MjvScene, points: list[np.ndarray], radius: float, rgba, start_idx: int = 0) -> int:
    idx = start_idx
    for p in points:
        idx = add_sphere_geom(scn, idx, p, radius, rgba)
    scn.ngeom = idx
    return idx


def wait_for_start(viewer, reset_state) -> bool:
    reset_state()
    print("  >> 按空格键开始仿真…")
    started = {"value": False}

    def on_key(keycode: int) -> None:
        if keycode == START_KEYCODE:
            started["value"] = True

    viewer.user_scn.ngeom = 0
    return started, on_key


def load_mujoco_scene(scene_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if scene_path.suffix == ".mjb":
        mj_model = mujoco.MjModel.from_binary_path(str(scene_path))
    else:
        mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    return mj_model, mujoco.MjData(mj_model)


def run_realtime_sleep(t0: float, sim_t: float) -> None:
    target = t0 + sim_t
    while time.time() < target:
        time.sleep(0.0005)
