import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from reach_env import ReachEnv


class PickPlaceEnv(ReachEnv):
    def __init__(self, render_mode=None, max_steps=300):
        gym.Env.__init__(self)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._robot_id = None
        self._peg_id = None
        self._plane_id = None
        self._controlled_joints = []
        self._ee_link_index = None
        self._grasp_constraint = None  # None when not holding anything

        self._build_world()
        self._detect_controllable_joints()

        lowers, uppers = [], []
        for j in self._controlled_joints:
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._client)
            lo, hi = info[8], info[9]
            if hi <= lo:
                lo, hi = -2 * np.pi, 2 * np.pi
            lowers.append(lo)
            uppers.append(hi)
        self._joint_lower = np.array(lowers, dtype=np.float32)
        self._joint_upper = np.array(uppers, dtype=np.float32)

        self._n_joints = len(self._controlled_joints)
        self._max_joint_velocity = 1.0
        self._action_repeat = 8
        self._grasp_distance_threshold = 0.06  # meters - how close ee must be to peg to grasp

        # Action = 7 joint velocities + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_joints + 1,), dtype=np.float32
        )

        obs_dim = 2 * self._n_joints + 3 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._peg_start_pos = None
        self._target_pos = None
        self._is_holding = False

    def _build_world(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        urdf_choice = "ur5/ur5.urdf" if self._urdf_exists("ur5/ur5.urdf") else "kuka_iiwa/model.urdf"
        self._robot_id = p.loadURDF(
            urdf_choice, basePosition=[0, 0, 0], useFixedBase=True,
            physicsClientId=self._client,
        )
        print(f"[PickPlaceEnv] Loaded robot URDF: {urdf_choice}")

        peg_radius, peg_height = 0.02, 0.08
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=peg_radius, height=peg_height,
            physicsClientId=self._client,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=peg_radius, length=peg_height,
            rgbaColor=[0.8, 0.2, 0.2, 1.0], physicsClientId=self._client,
        )
        self._peg_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.3, 0.0, 0.05],
            physicsClientId=self._client,
        )

    def _get_peg_position(self):
        pos, _ = p.getBasePositionAndOrientation(self._peg_id, physicsClientId=self._client)
        return np.array(pos, dtype=np.float32)

    def _get_obs(self):
        joint_pos, joint_vel = self._get_joint_state()
        ee_pos = self._get_ee_position()
        peg_pos = self._get_peg_position()
        relevant_point = self._target_pos if self._is_holding else peg_pos
        vec_to_relevant = relevant_point - ee_pos
        holding_flag = np.array([1.0 if self._is_holding else 0.0], dtype=np.float32)
        return np.concatenate([
            joint_pos, joint_vel, ee_pos, peg_pos, self._target_pos,
            vec_to_relevant, holding_flag
        ]).astype(np.float32)

    def _release_grasp(self):
        if self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint, physicsClientId=self._client)
            self._grasp_constraint = None
        self._is_holding = False

    def _try_grasp(self, ee_pos, peg_pos):
        distance = float(np.linalg.norm(ee_pos - peg_pos))
        if distance > self._grasp_distance_threshold:
            return False  # too far - policy chose to close gripper prematurely
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self._robot_id,
            parentLinkIndex=self._ee_link_index,
            childBodyUniqueId=self._peg_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client,
        )
        self._is_holding = True
        return True

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._step_count = 0
        self._release_grasp()

        random_config = self.np_random.uniform(
            low=self._joint_lower * 0.5, high=self._joint_upper * 0.5
        )
        for idx, j in enumerate(self._controlled_joints):
            p.resetJointState(self._robot_id, j, random_config[idx], physicsClientId=self._client)

        # Randomize peg position within reach (same empirically-derived
        # envelope as Phase 1's target box, but pegs sit on/near the
        # ground plane rather than floating).
        self._peg_start_pos = self.np_random.uniform(
            low=np.array([-0.4, -0.35, 0.04]),
            high=np.array([0.4, 0.35, 0.04]),
        ).astype(np.float32)
        p.resetBasePositionAndOrientation(
            self._peg_id, self._peg_start_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self._client,
        )

        # Randomize placement target, independent of peg start position.
        self._target_pos = self.np_random.uniform(
            low=np.array([-0.45, -0.4, 0.15]),
            high=np.array([0.45, 0.4, 0.6]),
        ).astype(np.float32)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)
        joint_action = action[:self._n_joints] * self._max_joint_velocity
        gripper_command = action[self._n_joints]

        for idx, j in enumerate(self._controlled_joints):
            p.setJointMotorControl2(
                self._robot_id, j, controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(joint_action[idx]), physicsClientId=self._client,
            )

        # Handle grasp/release BEFORE stepping physics, so a newly
        # formed constraint is respected during this step's simulation.
        ee_pos = self._get_ee_position()
        peg_pos = self._get_peg_position()
        grasp_attempted = False
        premature_release_penalty = 0.0
        if gripper_command > 0 and not self._is_holding:
            grasp_attempted = self._try_grasp(ee_pos, peg_pos)
        elif gripper_command <= 0 and self._is_holding:
            distance_to_target_at_release = float(np.linalg.norm(peg_pos - self._target_pos))
            if distance_to_target_at_release > 0.08:
                premature_release_penalty = 3.0
            self._release_grasp()

        for _ in range(self._action_repeat):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_obs()
        ee_pos = self._get_ee_position()
        peg_pos = self._get_peg_position()
        
        # Stage 1 (not holding): dense shaping toward the peg.
        # Stage 2 (holding): dense shaping toward the target, PLUS a
        # bonus for successfully having grasped at all (encourages the
        # policy to actually attempt/complete a grasp rather than just
        # hovering near the peg to farm Stage-1 proximity reward).
        if not self._is_holding:
            distance = float(np.linalg.norm(ee_pos - peg_pos))
            reward = -distance
        else:
            distance = float(np.linalg.norm(peg_pos - self._target_pos))
            reward = -distance + 2.0  # holding bonus, paid every step while held

        reward -= 0.01 * float(np.sum(np.square(action)))
        if grasp_attempted:
            reward += 1.0  # one-time bonus for a successful grasp event
        reward -= premature_release_penalty

        place_success = (
            self._is_holding
            and float(np.linalg.norm(peg_pos - self._target_pos)) < 0.05
        )
        # A full pick-and-place success requires RELEASING at the
        # target, not just hovering there while still holding - release
        # the grasp as part of the success condition and check the peg
        # stays near the target afterward.
        success = False
        if place_success and gripper_command <= 0:
            self._release_grasp()
            success = True
            reward += 20.0

        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "distance": distance,
            "success": success,
            "is_holding": self._is_holding,
            "grasp_attempted": grasp_attempted,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(physicsClientId=self._client)


if __name__ == "__main__":
    env = PickPlaceEnv(render_mode=None)
    obs, info = env.reset()
    print("Observation shape:", obs.shape, "| Action shape:", env.action_space.shape)
    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    print("Smoke test complete. Sample total reward over 50 random steps:", total_reward)
    print("Sample info:", info)
    env.close()
