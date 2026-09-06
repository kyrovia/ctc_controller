#include "ctc_control/pinocchio_model.hpp"

#include <cctype>
#include <stdexcept>
#include <string>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace ctc_control {
namespace {

bool ends_with_ci(const std::string& path, const char* suffix) {
  const std::size_t suffix_len = std::char_traits<char>::length(suffix);
  if (path.size() < suffix_len) {
    return false;
  }
  for (std::size_t i = 0; i < suffix_len; ++i) {
    const char a = static_cast<char>(
        std::tolower(static_cast<unsigned char>(path[path.size() - suffix_len + i])));
    const char b = static_cast<char>(
        std::tolower(static_cast<unsigned char>(suffix[i])));
    if (a != b) {
      return false;
    }
  }
  return true;
}

bool is_mjcf_path(const std::string& path) {
  return ends_with_ci(path, ".xml") || ends_with_ci(path, ".mjcf");
}

void build_model(const std::string& model_path, pinocchio::Model& model) {
  if (is_mjcf_path(model_path)) {
    pinocchio::mjcf::buildModel(model_path, model);
  } else {
    pinocchio::urdf::buildModel(model_path, model);
  }
  if (model.nv == 0) {
    throw std::invalid_argument("Model file produced an empty Pinocchio model: " + model_path);
  }
}

}  // namespace

PinocchioRobotModel PinocchioRobotModel::from_file(const std::string& model_path) {
  pinocchio::Model model;
  build_model(model_path, model);
  return PinocchioRobotModel(std::move(model));
}

PinocchioRobotModel::PinocchioRobotModel(pinocchio::Model model)
    : model_(std::move(model)), data_(model_) {}

int PinocchioRobotModel::nv() const { return model_.nv; }

VectorXd PinocchioRobotModel::rnea(
    const VectorXd& q, const VectorXd& dq, const VectorXd& qdd) const {
  return pinocchio::rnea(model_, data_, q, dq, qdd);
}

}  // namespace ctc_control
