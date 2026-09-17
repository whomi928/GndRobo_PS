# Part 1 — SLAM & Autonomous Navigation

**Inter IIT Tech Meet 15.0 — Prepathon PS: Ground Robotics**

This package implements autonomous SLAM-based mapping and point-to-point navigation on a holonomic (mecanum-wheel) mobile base, using ROS2 Jazzy and Gazebo Harmonic.

---

## 1. Platform & Stack

| Component | Choice |
|---|---|
| ROS2 distro | Jazzy |
| Simulator | Gazebo Harmonic (`gz sim` 8.11.0) |
| Robot base | Mecanum-wheel holonomic base, adapted from [`automaticaddison/yahboom_rosmaster`](https://github.com/automaticaddison/yahboom_rosmaster) |
| Drive controller | Custom `mecanum_drive_controller` (ros2_control plugin, ships with the base repo) |
| Localization (mapping) | `slam_toolbox` (online async mode) |
| Localization (navigation) | AMCL (`nav2_amcl::OmniMotionModel`) against a saved static map |
| Sensor fusion | `robot_localization` EKF, fusing wheel odometry + IMU yaw-rate |
| Navigation stack | Nav2 (planner, controller, BT navigator, recovery behaviors) |
| World | `cafe.world` |

The platform itself (mecanum base, sensors, low-level drive plugin) was treated as a given, per the PS scope note — our work is the SLAM/navigation stack layered on top of it, plus the sensor-fusion debugging described below.

---

## 2. Setup & Run Instructions

### One-time setup
```bash
mkdir -p ~/gr_ws/src && cd ~/gr_ws/src
git clone https://github.com/automaticaddison/yahboom_rosmaster.git
cd ~/gr_ws
sudo rosdep init      # if not already done
rosdep update
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build
source install/setup.bash
echo "source ~/gr_ws/install/setup.bash" >> ~/.bashrc
```

A symlink is required because the repo's bringup script hardcodes `~/ros2_ws` in one place:
```bash
ln -s ~/gr_ws ~/ros2_ws
```

### Run — Mapping (SLAM)
```bash
# Terminal 1: bring up Gazebo + robot + controllers in SLAM mode
bash ~/gr_ws/src/yahboom_rosmaster/yahboom_rosmaster_bringup/scripts/rosmaster_x3_navigation.sh slam

# Terminal 2: start the mapping node
source ~/gr_ws/install/setup.bash
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=/home/shaur/gr_ws/slam_config/mapper_params.yaml \
  use_sim_time:=true

# Terminal 3: drive the robot around to build the map
source ~/gr_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Once coverage looks complete in RViz, save the map:
ros2 run nav2_map_server map_saver_cli -f ~/gr_ws/my_cafe_map
```

### Run — Autonomous Navigation
```bash
# Bring up Gazebo + robot + Nav2 + AMCL against the saved map
bash ~/gr_ws/src/yahboom_rosmaster/yahboom_rosmaster_bringup/scripts/rosmaster_x3_navigation.sh
```
In RViz, use the **Nav2 Goal** tool to click-and-drag a target pose. The robot autonomously plans and drives to it, replanning around obstacles and recovering from stalls via Nav2's built-in behavior tree recoveries.

---

## 3. Deliverables in this folder

```
maps/
  my_cafe_map.yaml
  my_cafe_map.pgm
videos/
  mapping_demo.mp4        # slam_toolbox building the map while teleoperated
  navigation_demo.mp4      # autonomous point-to-point navigation via Nav2
yahboom_rosmaster/          # full modified ROS2 workspace source
README.md                   # this file
```

---

## 4. Sensor Selection & Fusion Justification

- **2D LiDAR** is the primary mapping sensor, feeding `/scan` directly into `slam_toolbox`'s scan-matching pipeline — this is the sensor the PS specifies as primary, and it's the only sensor that directly observes obstacle geometry.
- **Wheel odometry** from the mecanum controller provides continuous, high-rate (50 Hz) linear and angular velocity estimates, but drifts over time and has no absolute heading reference.
- **IMU** contributes only angular velocity around Z (yaw rate) to the EKF. We deliberately do **not** fuse raw IMU linear acceleration — see the debugging section below for why.
- **EKF (`robot_localization`)** fuses wheel-odometry velocity (x, y, yaw-rate) with IMU yaw-rate into a single `odom → base_footprint` estimate, which is what both `slam_toolbox` and Nav2's costmaps consume as ground truth for the robot's local pose.

This is the standard REP-105-compliant frame setup: `map` (global, corrected by AMCL/SLAM) → `odom` (continuous, EKF-fused) → `base_footprint` (robot body).

---

## 5. Debugging Notes (approach & reasoning)

The cloned base repo required several fixes before the SLAM/localization pipeline was usable. These are documented here since they reflect the actual engineering work behind this submission, not just a working black box:

1. **Hardcoded workspace path.** The bringup script and a controller config referenced `/home/ubuntu/ros2_ws/...` regardless of the actual workspace location. Fixed by symlinking `~/ros2_ws → ~/gr_ws` and patching the remaining hardcoded map path in the launch script.

2. **Missing rosdep init.** `urdf_tutorial` and other transitive dependencies failed to resolve until `rosdep init && rosdep update` was run before `rosdep install`.

3. **SLAM mode published no map.** The repo's `slam:=True` launch argument only started a `lifecycle_manager_slam` node managing `map_saver` — it never actually launched `slam_toolbox` itself. Fixed by launching `slam_toolbox`'s `online_async_launch.py` manually alongside the existing bringup.

4. **Duplicate `odom → base_footprint` TF broadcasters.** With `enable_odom_tf:=true`, both `mecanum_drive_controller` and `ekf_filter_node` were independently broadcasting the same transform, corrupting tf2's interpolation buffer. Fixed by setting `enable_odom_tf:=false` on the controller so `robot_localization`'s EKF is the sole TF broadcaster for that link (matching REP-105 guidance: only one node should publish `odom → base_link`).

5. **Zero-covariance odometry.** The controller published `pose_covariance_diagonal` / `twist_covariance_diagonal` as all zeros. Zero covariance is interpreted by the EKF as infinite measurement confidence, which is mathematically unstable. Set both to small non-zero diagonal values (`0.001`–`0.01`).

6. **EKF divergence from raw IMU acceleration fusion.** Even after fixing (4) and (5), the EKF's position estimate still diverged to effectively infinite values while the robot was stationary, with covariance exploding to `~10^34`. Root cause: `ekf.yaml`'s `imu0_config` fused raw linear acceleration (`ax`, `ay`), which requires double integration to contribute to position — any small bias or simulated sensor noise compounds unbounded over time. Fixed by disabling acceleration fusion entirely, keeping only IMU yaw-rate (`vyaw`) fused alongside wheel-odometry velocities. This is a known instability pattern in `robot_localization` and the fix follows their own documented best practice of preferring velocity-level fusion over raw acceleration for ground robots.

After these fixes, `odom → base_footprint` TF was verified stable and near-zero at rest, `slam_toolbox` produced a clean, consistent occupancy grid, and AMCL localized correctly against the saved map for autonomous Nav2 navigation.

---

## 6. Known Limitations

- Full navigation stack was tested against a single mapped world (`cafe.world`); robustness across other environments not evaluated.
- No dynamic obstacle avoidance testing was performed beyond Nav2's default local costmap behavior — dynamic/moving obstacles were not introduced into the world.
- Map quality depends on manual teleoperation coverage rather than an automated exploration policy (e.g. frontier exploration), which is out of scope for this PS.
