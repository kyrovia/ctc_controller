#include "ctc_control/pinocchio_model.hpp"

#include <stdexcept>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace ctc_control {
//解析urdf
PinocchioRobotModel PinocchioRobotModel::from_urdf(const std::string& urdf_path) {
  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdf_path, model);
  return PinocchioRobotModel(std::move(model));
}
//构造函数初始化
PinocchioRobotModel::PinocchioRobotModel(pinocchio::Model model)
    : model_(std::move(model)), data_(model_) {}

int PinocchioRobotModel::nv() const { return model_.nv; }
//rnea计算逆动力学
VectorXd PinocchioRobotModel::rnea(
    const VectorXd& q, const VectorXd& dq, const VectorXd& qdd) const {
  return pinocchio::rnea(model_, data_, q, dq, qdd);
}

}  // namespace ctc_control
