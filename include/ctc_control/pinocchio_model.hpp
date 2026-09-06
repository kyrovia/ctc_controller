#pragma once

#include <string>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "ctc_control/robot_model.hpp"

namespace ctc_control {

/// Pinocchio实现的机器人模型
class PinocchioRobotModel final : public RobotModel {
public:
  
  static PinocchioRobotModel from_urdf(const std::string& urdf_path);

  explicit PinocchioRobotModel(pinocchio::Model model);

  [[nodiscard]] int nv() const override;

  [[nodiscard]] VectorXd rnea(
      const VectorXd& q, const VectorXd& dq, const VectorXd& qdd) const override;

private:
  pinocchio::Model model_;
  mutable pinocchio::Data data_;
};

}  // namespace ctc_control
