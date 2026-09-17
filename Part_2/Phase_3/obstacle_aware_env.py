import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from pick_place_env import PickPlaceEnv


class ObstacleAwareEnv(PickPlaceEnv):
    def __init__(self, render_mode=None, max_steps=300, n_obstacles=3):
        self._n_obstacles = n_obstacles
        self._obstacle_ids = []
        self._obstacle_half_extent = 0.035  # small boxes, ~7cm cubes
        self._min_clearance = 0.12  # min distance an obstacle must keep from peg/target at spawn

        super().__init__(render_mode=render_mode, max_steps=max_steps)

        obs_dim = self.observation_space.shape[0] + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._collision_this_step = False
        self._episode_collision_steps = 0

    def _build_world(self):
        # Reuse Phase 2's robot + peg setup, then add obstacles.
        super()._build_world()
        self._obstacle_ids = []
        for _ in range(self._n_obstacles):
            col_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self._obstacle_half_extent] * 3,
                physicsClientId=self._client,
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self._obstacle_half_extent] * 3,
                rgbaColor=[0.2, 0.2, 0.8, 1.0],
                physicsClientId=self._client,
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,  # static (mass=0 -> fixed in place, never moves)
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[10, 10, 10],  # placeholder, positioned properly in reset()
                physicsClientId=self._client,
            )
            self._obstacle_ids.append(obstacle_id)

    def _sample_obstacle_positions(self):
        positions = []
        attempts = 0
        while len(positions) < self._n_obstacles and attempts < 200:
            attempts += 1
            candidate = self.np_random.uniform(
                low=np.array([-0.4, -0.35, 0.1]),
                high=np.array([0.4, 0.35, 0.5]),
            ).astype(np.float32)
            if np.linalg.norm(candidate - self._peg_start_pos) < self._min_clearance:
                continue
            if np.linalg.norm(candidate - self._target_pos) < self._min_clearance:
                continue
            if any(np.linalg.norm(candidate - p_) < self._min_clearance for p_ in positions):
                continue  # keep obstacles from overlapping each other too
            positions.append(candidate)
        # If we couldn't find enough valid spots (rare, cramped workspace),
        # pad with far-away/inert positions rather than crashing.
        while len(positions) < self._n_obstacles:
            positions.append(np.array([0.9, 0.9, 0.9], dtype=np.float32))
        return positions

    def _get_nearest_obstacle_info(self, ee_pos):
        if len(self._obstacle_ids) == 0:
            return np.array([10.0, 10.0, 10.0], dtype=np.float32)
        obstacle_positions = [
            np.array(p.getBasePositionAndOrientation(oid, physicsClientId=self._client)[0])
            for oid in self._obstacle_ids
        ]
        distances = [np.linalg.norm(ee_pos - pos) for pos in obstacle_positions]
        nearest_idx = int(np.argmin(distances))
        return obstacle_positions[nearest_idx] - ee_pos

    def _check_collisions(self):
        for obstacle_id in self._obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self._robot_id, bodyB=obstacle_id, physicsClientId=self._client
            )
            if len(contacts) > 0:
                return True
        return False

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._episode_collision_steps = 0
        self._collision_this_step = False

        obstacle_positions = self._sample_obstacle_positions()
        for oid, pos in zip(self._obstacle_ids, obstacle_positions):
            p.resetBasePositionAndOrientation(
                oid, pos.tolist(), [0, 0, 0, 1], physicsClientId=self._client
            )

        obs = self._get_obs()
        return obs, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        ee_pos = self._get_ee_position()
        nearest_obstacle_vec = self._get_nearest_obstacle_info(ee_pos)
        collision_flag = np.array(
            [1.0 if self._collision_this_step else 0.0], dtype=np.float32
        )
        return np.concatenate([base_obs, nearest_obstacle_vec, collision_flag]).astype(np.float32)

    def step(self, action):
        # Run Phase 2's full step logic (motor commands, grasp handling,
        # physics stepping, staged reward) via the parent class, then
        # layer collision detection/penalty on top.
        obs, reward, terminated, truncated, info = super().step(action)

        self._collision_this_step = self._check_collisions()
        if self._collision_this_step:
            self._episode_collision_steps += 1
            reward -= 1.0

        obs = self._get_obs()  # rebuild obs now that collision_flag is current

        info["collision"] = self._collision_this_step
        info["episode_collision_steps"] = self._episode_collision_steps

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = ObstacleAwareEnv(render_mode=None, n_obstacles=3)
    obs, info = env.reset()
    print("Observation shape:", obs.shape, "| Action shape:", env.action_space.shape)
    total_reward = 0.0
    total_collision_steps = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["collision"]:
            total_collision_steps += 1
        if terminated or truncated:
            obs, info = env.reset()
    print("Smoke test complete. Sample total reward over 50 random steps:", total_reward)
    print("Collision steps observed in smoke test:", total_collision_steps)
    print("Sample info:", info)
    env.close()
