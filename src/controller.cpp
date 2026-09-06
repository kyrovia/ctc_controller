#include "ctc_control/controller.hpp"

#include <stdexcept>

namespace ctc_control {
namespace {

bool same_size(const VectorXd& a, int n) { return a.size() == n; }

VectorXd saturate_abs(const VectorXd& tau, const VectorXd& limit) {
  return tau.cwiseMax(-limit).cwiseMin(limit);
}

VectorXd saturate_rate(const VectorXd& tau, const VectorXd& prev, double max_delta) {
  const VectorXd delta = (tau - prev).cwiseMax(-max_delta).cwiseMin(max_delta);
  return prev + delta;
}

}  // namespace

CTCController::CTCController(
    std::shared_ptr<const RobotModel> model, Gains gains, Safety safety)
    : model_(std::move(model)), gains_(std::move(gains)), safety_(std::move(safety)) {
  if (!model_) {
    throw std::invalid_argument("CTCController: model is null");
  }
  check_sizes(gains_);
  reset();
}

void CTCController::check_sizes(const Gains& gains) const {
  const int n = model_->nv();
  if (gains.kp.size() != n || gains.kd.size() != n) {
    throw std::invalid_argument("Gains kp/kd must match model nv");
  }
  if (gains.ki.size() != 0 && gains.ki.size() != n) {
    throw std::invalid_argument("Gains ki must be empty or match model nv");
  }
  if (safety_.tau_abs_max.size() != 0 && safety_.tau_abs_max.size() != n) {
    throw std::invalid_argument("Safety.tau_abs_max must be empty or match model nv");
  }
}

void CTCController::reset() {
  const int n = model_->nv();
  e_int_ = VectorXd::Zero(n);
  tau_prev_ = VectorXd::Zero(n);
  has_prev_ = false;
}

VectorXd CTCController::step(
    const JointState& state, const TrajectoryRef& ref, double dt) {
  const int n = model_->nv();
  const bool sizes_ok = same_size(state.q, n) && same_size(state.dq, n) && same_size(ref.qd, n) &&
                        same_size(ref.dqd, n) && same_size(ref.ddqd, n);
  if (!sizes_ok) {
    return tau_prev_;
  }

  const VectorXd err = ref.qd - state.q;
  const VectorXd derr = ref.dqd - state.dq;

  if (gains_.ki.size() == n && dt > 0.0) {
    e_int_ += err * dt;
    if (safety_.integral_limit > 0.0) {
      e_int_ = saturate_abs(e_int_, VectorXd::Constant(n, safety_.integral_limit));
    }
  }

  VectorXd qdd_cmd = ref.ddqd + gains_.kp.cwiseProduct(err) + gains_.kd.cwiseProduct(derr);
  if (gains_.ki.size() == n) {
    qdd_cmd += gains_.ki.cwiseProduct(e_int_);
  }

  VectorXd tau = model_->rnea(state.q, state.dq, qdd_cmd);

  if (safety_.hold_on_nonfinite && !tau.allFinite()) {
    return tau_prev_;
  }
  if (safety_.tau_abs_max.size() == n) {
    tau = saturate_abs(tau, safety_.tau_abs_max);
  }
  if (has_prev_ && safety_.max_delta_tau > 0.0) {
    tau = saturate_rate(tau, tau_prev_, safety_.max_delta_tau);
  }

  tau_prev_ = tau;
  has_prev_ = true;
  return tau;
}

}  // namespace ctc_control
