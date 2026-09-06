#pragma once

#include "ctc_control/types.hpp"

namespace ctc_control {

///机器人模型抽象接口，支持多种后端实现
class RobotModel {
public:
  virtual ~RobotModel() = default;

  [[nodiscard]] virtual int nv() const = 0;

  /// Inverse dynamics: tau = M(q) qdd + C(q, dq) dq + g(q).
  [[nodiscard]] virtual VectorXd rnea(
      const VectorXd& q, const VectorXd& dq, const VectorXd& qdd) const = 0;
};

}  // namespace ctc_control
