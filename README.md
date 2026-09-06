# ctc_controller

关节空间 **计算力矩控制（CTC / inverse dynamics）** 的 C++ 工程库。不含 ROS。动力学来自 **URDF / MJCF + Pinocchio**。

## 编译

```bash
conda activate gct
cd /home/l/ctc_controller
cmake -S . -B build
cmake --build build -j
cmake --build build --target ctc_mujoco_scene ctc_mujoco_scene_obstacle
```

依赖：Eigen3、Pinocchio、MuJoCo Python、matplotlib（力矩曲线 demo）。

## Demo 1：空间圆轨迹跟踪（MuJoCo 显示参考圆）

```bash
cmake --build build --target ur5e_circle_demo
# 或
python examples/ur5e_figure8_demo.py --real-time
```

- 绿色点云：笛卡尔空间参考圆
- 橙色轨迹：实际末端路径
- 黄色球：当前末端位置
- 按 **空格** 开始

## Demo 2：障碍物碰撞 + 力矩限幅曲线

```bash
cmake --build build --target ur5e_obstacle_torque_demo
# 或
python examples/ur5e_obstacle_torque_demo.py --real-time
```

- 红色方块：障碍物
- 机械臂 CTC 跟踪一条会穿过障碍物的参考轨迹
- 参考轨迹采用五次最小加加速度插值，避免接近过程抖动
- 实际碰到障碍物后继续顶压，力矩才逐渐达到限幅
- 旁边 matplotlib 窗口实时显示 6 关节力矩
- 蓝色 = 限幅后输出，橙色 = 限幅前 raw
- 红色虚线 = ±max_tau
- 默认演示限幅为 `25,20,20,5,5,5 Nm`，可通过 `--max-tau` 修改

## 自动化测试

```bash
ctest --test-dir build --output-on-failure
```

UR5e 模型路径：`/home/l/gravity_comp/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml`
