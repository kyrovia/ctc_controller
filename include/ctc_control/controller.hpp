#pragma once

#include <memory>

#include "ctc_control/robot_model.hpp"
#include "ctc_control/types.hpp"

namespace ctc_control {


class CTCController {
public:
  CTCController(
      std::shared_ptr<const RobotModel> model, Gains gains, Safety safety = {});

  void reset();

  [[nodiscard]] VectorXd step(const JointState& state, const TrajectoryRef& ref, double dt);

private:
  void check_sizes(const Gains& gains) const;

  std::shared_ptr<const RobotModel> model_;
  Gains gains_;
  Safety safety_;
  VectorXd e_int_;
  VectorXd tau_prev_;
  bool has_prev_{false};
};

}  // namespace ctc_control
