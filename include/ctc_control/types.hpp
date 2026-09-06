#pragma once

#include <Eigen/Dense>

namespace ctc_control {

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

//作为rnea的部分输入
struct JointState {
  VectorXd q;
  VectorXd dq;
};

struct TrajectoryRef {
  VectorXd qd;
  VectorXd dqd;
  VectorXd ddqd;
};

struct Gains {
  VectorXd kp;
  VectorXd kd;
  VectorXd ki;  //PD+前馈后仍有误差开启，默认关闭
};

struct Safety {
  //每个关节的力矩最大限副
  VectorXd tau_abs_max;
  //每个关节的力矩最大变化
  double max_delta_tau{20.0};
  bool hold_on_nonfinite{true};
  //积分误差最大限副
  double integral_limit{2.0};
};

}  // namespace ctc_control
