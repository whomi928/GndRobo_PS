import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class ReachEnv(gym.Env):
    

    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None, max_steps=200):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._robot_id = None
        self._controlled_joints = []   # filled in by _build_world()
        self._ee_link_index = None     # filled in by _build_world()
        self._n_joints = None

        self._build_world()  # loads robot_id, THEN detects joints below

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
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_joints,), dtype=np.float32
        )
        self._max_joint_velocity = 1.0  # rad/s, applied after scaling action
        self._action_repeat = 8

        obs_dim = 2 * self._n_joints + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_pos = None

    def _build_world(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        urdf_choice = "ur5/ur5.urdf" if self._urdf_exists("ur5/ur5.urdf") else "kuka_iiwa/model.urdf"
        self._robot_id = p.loadURDF(
            urdf_choice,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        print(f"[ReachEnv] Loaded robot URDF: {urdf_choice}")

    @staticmethod
    def _urdf_exists(rel_path):
        import os
        return os.path.exists(os.path.join(pybullet_data.getDataPath(), rel_path))

    def _detect_controllable_joints(self):
        n = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        controlled = []
        for j in range(n):
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._client)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                controlled.append(j)
        if not controlled:
            raise RuntimeError(
                "No revolute joints found on loaded robot - check the URDF."
            )
        self._controlled_joints = controlled
        self._ee_link_index = controlled[-1]
        joint_names = [
            p.getJointInfo(self._robot_id, j, physicsClientId=self._client)[1].decode("utf-8")
            for j in controlled
        ]
        print(f"[ReachEnv] Detected {len(controlled)} controllable joints: {joint_names}")
        print(f"[ReachEnv] Using link index {self._ee_link_index} as end-effector.")

    def _get_joint_state(self):
        positions, velocities = [], []
        for j in self._controlled_joints:
            state = p.getJointState(self._robot_id, j, physicsClientId=self._client)
            positions.append(state[0])
            velocities.append(state[1])
        return np.array(positions, dtype=np.float32), np.array(velocities, dtype=np.float32)

    def _get_ee_position(self):
        link_state = p.getLinkState(
            self._robot_id, self._ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        return np.array(link_state[0], dtype=np.float32)  # world position (x, y, z)

    def _get_obs(self):
        joint_pos, joint_vel = self._get_joint_state()
        ee_pos = self._get_ee_position()
        vec_to_target = self._target_pos - ee_pos
        return np.concatenate([joint_pos, joint_vel, ee_pos, self._target_pos, vec_to_target]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        random_config = self.np_random.uniform(
            low=self._joint_lower * 0.5, high=self._joint_upper * 0.5
        )
        for idx, j in enumerate(self._controlled_joints):
            p.resetJointState(self._robot_id, j, random_config[idx], physicsClientId=self._client)

        self._target_pos = self.np_random.uniform(
            low=np.array([-0.45, -0.4, 0.15]),
            high=np.array([0.45, 0.4, 0.9]),
        ).astype(np.float32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0) * self._max_joint_velocity

        for idx, j in enumerate(self._controlled_joints):
            p.setJointMotorControl2(
                self._robot_id, j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(action[idx]),
                physicsClientId=self._client,
            )
        for _ in range(self._action_repeat):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_obs()
        ee_pos = self._get_ee_position()
        distance = float(np.linalg.norm(ee_pos - self._target_pos))

        # Dense negative-distance term: every step, the agent gets
        # feedback proportional to how far it still is - this is the
        # gradient signal that makes learning tractable from step one.
        reward = -distance

        # Small action-magnitude penalty discourages jerky, wasteful
        # motion and invalid/singular-looking flailing (partially
        # addresses the PS's "avoid invalid/singular configurations").
        reward -= 0.01 * float(np.sum(np.square(action)))

        success = distance < 0.03  # 3cm tolerance - tune per your grading needs
        if success:
            reward += 10.0  # sparse bonus on top of the dense shaping

        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps)

        info = {"distance": distance, "success": success}
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(physicsClientId=self._client)


if __name__ == "__main__":
    # training run that might take 30+ minutes.
    env = ReachEnv(render_mode=None)
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
