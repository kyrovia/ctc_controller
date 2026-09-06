cmake_minimum_required(VERSION 3.16)
project(ctc_control VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

list(APPEND CMAKE_PREFIX_PATH
  "$ENV{CONDA_PREFIX}"
  "/home/l/miniconda3/envs/gct")
find_package(pinocchio CONFIG REQUIRED)

option(CTC_ENABLE_MUJOCO "Build UR5e MuJoCo viewer and visual tests" ON)

set(CTC_UR5E_MJCF_DEFAULT
  "/home/l/gravity_comp/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
if(EXISTS "${CTC_UR5E_MJCF_DEFAULT}")
  set(CTC_UR5E_MJCF "${CTC_UR5E_MJCF_DEFAULT}" CACHE FILEPATH "Menagerie UR5e MJCF path")
else()
  set(CTC_UR5E_MJCF "" CACHE FILEPATH "Menagerie UR5e MJCF path")
endif()

add_library(ctc_control
  src/controller.cpp
  src/pinocchio_model.cpp
)
add_library(ctc_control::ctc_control ALIAS ctc_control)

target_include_directories(ctc_control
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ctc_control PUBLIC Eigen3::Eigen pinocchio::pinocchio)
target_compile_options(ctc_control PRIVATE -Wall -Wextra -Wpedantic)

if(CTC_ENABLE_MUJOCO)
  if(NOT CTC_UR5E_MJCF)
    message(FATAL_ERROR
      "UR5e MJCF not found. Set -DCTC_UR5E_MJCF=/path/to/ur5e.xml or install Menagerie UR5e.")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import mujoco"
    RESULT_VARIABLE _ctc_mujoco_import
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT _ctc_mujoco_import EQUAL 0)
    message(FATAL_ERROR
      "Python package 'mujoco' not found for ${Python3_EXECUTABLE}. "
      "Activate conda env gct or pass -DCTC_ENABLE_MUJOCO=OFF.")
  endif()

  set(CTC_MUJOCO_SCENE_PATH "${CMAKE_BINARY_DIR}/models/ur5e_menagerie_torque.mjb")
  set(CTC_MUJOCO_SCENE_OBSTACLE_PATH "${CMAKE_BINARY_DIR}/models/ur5e_menagerie_torque_obstacle.mjb")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/models")

  add_custom_command(
    OUTPUT "${CTC_MUJOCO_SCENE_PATH}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/build_menagerie_torque_scene.py"
            --mjcf "${CTC_UR5E_MJCF}"
            --output "${CTC_MUJOCO_SCENE_PATH}"
    DEPENDS "${CTC_UR5E_MJCF}" "${CMAKE_SOURCE_DIR}/tools/build_menagerie_torque_scene.py"
    COMMENT "Building Menagerie UR5e torque scene (ur5e_menagerie_torque.mjb)"
    VERBATIM
  )
  add_custom_target(ctc_mujoco_scene DEPENDS "${CTC_MUJOCO_SCENE_PATH}")

  add_custom_command(
    OUTPUT "${CTC_MUJOCO_SCENE_OBSTACLE_PATH}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/build_menagerie_torque_scene.py"
            --mjcf "${CTC_UR5E_MJCF}"
            --output "${CTC_MUJOCO_SCENE_OBSTACLE_PATH}"
            --obstacle
    DEPENDS "${CTC_UR5E_MJCF}" "${CMAKE_SOURCE_DIR}/tools/build_menagerie_torque_scene.py"
    COMMENT "Building Menagerie UR5e torque scene with obstacle"
    VERBATIM
  )
  add_custom_target(ctc_mujoco_scene_obstacle DEPENDS "${CTC_MUJOCO_SCENE_OBSTACLE_PATH}")

  add_custom_target(ur5e_circle_demo
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/examples/ur5e_figure8_demo.py"
            --real-time
    DEPENDS ctc_mujoco_scene
    USES_TERMINAL
    COMMENT "Launch UR5e Cartesian circle tracking demo"
  )
  add_custom_target(ur5e_obstacle_torque_demo
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/examples/ur5e_obstacle_torque_demo.py"
            --real-time
    DEPENDS ctc_mujoco_scene_obstacle
    USES_TERMINAL
    COMMENT "Launch UR5e obstacle collision + torque plot demo"
  )

  add_custom_target(ur5e_ctc_viewer
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/examples/ur5e_ctc_viewer.py"
            --mode track --real-time
    DEPENDS ctc_mujoco_scene
    USES_TERMINAL
    COMMENT "Launch UR5e CTC viewer: sinusoid tracking"
  )
  add_custom_target(ur5e_ctc_viewer_hold
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/examples/ur5e_ctc_viewer.py"
            --mode hold --real-time
    DEPENDS ctc_mujoco_scene
    USES_TERMINAL
    COMMENT "Launch UR5e CTC viewer: hold at home"
  )
  add_custom_target(ur5e_ctc_viewer_compare
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/examples/ur5e_ctc_viewer.py"
            --mode compare --real-time
    DEPENDS ctc_mujoco_scene
    USES_TERMINAL
    COMMENT "Launch UR5e CTC viewer: no-ctc -> hold -> track"
  )

  enable_testing()
  add_custom_target(ur5e_ctc_headless_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/examples/ur5e_ctc_viewer.py"
            --headless --mode track --duration 2.0
    DEPENDS ctc_mujoco_scene
    COMMENT "Headless UR5e CTC tracking validation"
  )
  add_test(
    NAME ur5e_ctc_track_headless
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target ur5e_ctc_headless_test
  )
  set_tests_properties(ur5e_ctc_track_headless PROPERTIES
    ENVIRONMENT "MUJOCO_GL=egl"
  )
endif()
