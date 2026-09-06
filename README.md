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

## Demo 1：高速圆轨迹跟踪与精度对比

```bash
cmake --build build --target ur5e_circle_demo
# 或
python examples/ur5e_figure8_demo.py --real-time
# 只运行定量基准，不打开窗口
python examples/ur5e_figure8_demo.py --headless
```

- 默认 TCP 平均速度为 `0.6 m/s`，半径 `0.08 m` 时周期约 `0.84 s`
- 自动对比完整 CTC 与纯关节位置 PID（不使用动力学或重力补偿）
- 输出关节 RMSE/最大误差、末端 RMSE/最大误差和力矩饱和率
- 绿色点云：笛卡尔空间参考圆
- 橙色轨迹：实际末端路径
- 黄色球：当前末端位置
- 按 **空格** 开始

![Demo 1：高速圆轨迹跟踪](assets/demo1_high_speed_circle.gif)

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
- 默认限幅为 UR5e 关节上限：`150,150,150,28,28,28 Nm`（J1–J3 大关节 150，J4–J6 腕部 28）
- 可通过 `--max-tau` 覆盖；例如 `--max-tau 25,20,20,5,5,5` 可更容易观察饱和

![Demo 2：障碍物碰撞与力矩限幅](assets/demo2_torque_limit.gif)

## 自动化测试

```bash
ctest --test-dir build --output-on-failure
```

UR5e 模型路径：`/home/l/gravity_comp/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml`
