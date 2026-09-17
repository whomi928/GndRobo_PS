import numpy as np
import pybullet as p

import obstacle_aware_env as _oae
from obstacle_aware_env import ObstacleAwareEnv
from pick_place_env import TABLE_TOP_Z

N_OBSTACLES = getattr(_oae, "N_OBSTACLES", 3)
MIN_OBSTACLE_CLEARANCE = getattr(_oae, "MIN_OBSTACLE_CLEARANCE", 0.10)
OBSTACLE_HALF_EXTENTS = getattr(_oae, "OBSTACLE_HALF_EXTENTS", [0.03, 0.03, 0.06])

BOARD_THICKNESS = 0.02
BOARD_OUTER_HALF = 0.12      
HOLE_HALF_SIZE = 0.025       
INSERTION_DEPTH = 0.04       
POS_TOL = 0.012             
ANG_TOL = 0.15              
HOVER_MARGIN = 0.10        


class PegInsertEnv(ObstacleAwareEnv):
    def __init__(self, render_mode=None, max_steps=300, n_obstacles=N_OBSTACLES):
        self._board_id = None
        self._hole_xy = np.zeros(2, dtype=np.float32)
        self._board_top_z = TABLE_TOP_Z  # board mounted at the same reachable height as the pickup table
        self._insertion_started = False
        self._tolerance_violated_during_insertion = False

        super().__init__(render_mode=render_mode, max_steps=max_steps, n_obstacles=n_obstacles)

        self._disable_place_success = True

        from gymnasium import spaces
        base_obs_dim = self.observation_space.shape[0]
        # + hole_xyz(3) + peg_tilt_from_vertical(1) + insertion_depth_progress(1)
        new_obs_dim = base_obs_dim + 3 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32
        )

    def _build_world(self):
        super()._build_world()

        strip_half_thickness = (BOARD_OUTER_HALF - HOLE_HALF_SIZE) / 2.0
        strip_offset = HOLE_HALF_SIZE + strip_half_thickness

        # (halfExtents, localPosition) for each of the 4 strips: top, bottom, left, right
        strips = [
            ([BOARD_OUTER_HALF, strip_half_thickness, BOARD_THICKNESS / 2.0], [0, strip_offset, 0]),
            ([BOARD_OUTER_HALF, strip_half_thickness, BOARD_THICKNESS / 2.0], [0, -strip_offset, 0]),
            ([strip_half_thickness, HOLE_HALF_SIZE, BOARD_THICKNESS / 2.0], [-strip_offset, 0, 0]),
            ([strip_half_thickness, HOLE_HALF_SIZE, BOARD_THICKNESS / 2.0], [strip_offset, 0, 0]),
        ]
        half_extents_list = [s[0] for s in strips]
        positions_list = [s[1] for s in strips]

        col_ids = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX] * 4,
            halfExtents=half_extents_list,
            collisionFramePositions=positions_list,
            physicsClientId=self._client,
        )
        vis_ids = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * 4,
            halfExtents=half_extents_list,
            visualFramePositions=positions_list,
            rgbaColors=[[0.3, 0.3, 0.9, 1.0]] * 4,
            physicsClientId=self._client,
        )
        self._board_id = p.createMultiBody(
            baseMass=0.0,  # static - the board is a fixed fixture, not a manipulated object
            baseCollisionShapeIndex=col_ids,
            baseVisualShapeIndex=vis_ids,
            basePosition=[1.0, 1.0, self._board_top_z - BOARD_THICKNESS / 2.0],  # parked initially
            physicsClientId=self._client,
        )

    def _sample_hole_and_obstacles(self):
        hole_xy = self.np_random.uniform(
            low=np.array([-0.35, -0.3]), high=np.array([0.35, 0.3]),
        ).astype(np.float32)

        positions = []
        attempts = 0
        while len(positions) < self._n_obstacles and attempts < 200:
            attempts += 1
            xy = self.np_random.uniform(
                low=np.array([-0.35, -0.3]), high=np.array([0.35, 0.3]),
            )
            candidate = np.array([xy[0], xy[1], TABLE_TOP_Z + OBSTACLE_HALF_EXTENTS[2]], dtype=np.float32)
            too_close_to_peg = np.linalg.norm(candidate[:2] - self._peg_start_pos[:2]) < MIN_OBSTACLE_CLEARANCE
            too_close_to_hole = np.linalg.norm(candidate[:2] - hole_xy) < MIN_OBSTACLE_CLEARANCE
            too_close_to_other = any(
                np.linalg.norm(candidate[:2] - other[:2]) < MIN_OBSTACLE_CLEARANCE
                for other in positions
            )
            if not (too_close_to_peg or too_close_to_hole or too_close_to_other):
                positions.append(candidate)
        while len(positions) < self._n_obstacles:
            positions.append(np.array([1.0, 1.0, TABLE_TOP_Z + OBSTACLE_HALF_EXTENTS[2]], dtype=np.float32))

        return hole_xy, np.array(positions, dtype=np.float32)

    def reset(self, seed=None, options=None):
        from pick_place_env import PickPlaceEnv
        obs, info = PickPlaceEnv.reset(self, seed=seed, options=options)

        self._insertion_started = False
        self._tolerance_violated_during_insertion = False

        self._hole_xy, self._obstacle_positions = self._sample_hole_and_obstacles()
        for body_id, pos in zip(self._obstacle_ids, self._obstacle_positions):
            p.resetBasePositionAndOrientation(
                body_id, pos.tolist(), [0, 0, 0, 1], physicsClientId=self._client,
            )
        p.resetBasePositionAndOrientation(
            self._board_id,
            [self._hole_xy[0], self._hole_xy[1], self._board_top_z - BOARD_THICKNESS / 2.0],
            [0, 0, 0, 1], physicsClientId=self._client,
        )
        self._collision_count_this_ep = 0

        self._target_pos = np.array(
            [self._hole_xy[0], self._hole_xy[1], self._board_top_z + HOVER_MARGIN],
            dtype=np.float32,
        )

        return self._get_obs(), info

    def _get_peg_tilt_and_tip(self, peg_pos):
        _, orn = p.getBasePositionAndOrientation(self._peg_id, physicsClientId=self._client)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        local_z_world = rot_matrix[:, 2]  # peg's own z-axis expressed in world frame
        tilt = float(np.arccos(np.clip(np.dot(local_z_world, [0, 0, 1]), -1.0, 1.0)))
        tip_z = float(peg_pos[2] - (self._peg_height / 2.0) * local_z_world[2])
        return tilt, tip_z

    def _get_obs(self):
        base_obs = super()._get_obs()  # ObstacleAwareEnv's obs
        peg_pos = self._get_peg_position()
        tilt, tip_z = self._get_peg_tilt_and_tip(peg_pos)
        depth_progress = max(0.0, self._board_top_z - tip_z)
        hole_xyz = np.array([self._hole_xy[0], self._hole_xy[1], self._board_top_z], dtype=np.float32)
        extra = np.concatenate([hole_xyz, [tilt], [depth_progress]]).astype(np.float32)
        return np.concatenate([base_obs, extra]).astype(np.float32)

    def step(self, action):
        obs, reward, _parent_terminated, truncated, info = super().step(action)

        peg_pos = self._get_peg_position()
        tilt, tip_z = self._get_peg_tilt_and_tip(peg_pos)
        xy_offset = float(np.linalg.norm(peg_pos[:2] - self._hole_xy))
        depth_progress = max(0.0, self._board_top_z - tip_z)

        # same way obstacle collisions are counted.
        board_contacts = p.getContactPoints(
            bodyA=self._peg_id, bodyB=self._board_id, physicsClientId=self._client,
        ) + p.getContactPoints(
            bodyA=self._robot_id, bodyB=self._board_id, physicsClientId=self._client,
        )
        board_collision = len(board_contacts) > 0
        if board_collision:
            self._collision_count_this_ep += 1
            reward -= 2.0

        # held and has entered the vicinity of the hole.
        if self._is_holding:
            near_hole = xy_offset < POS_TOL * 3  # generous zone to start tracking, tighter for success
            if near_hole and tip_z < self._board_top_z + 0.02:
                self._insertion_started = True

            if self._insertion_started:
                tolerance_ok = (xy_offset < POS_TOL) and (tilt < ANG_TOL)
                if not tolerance_ok:
                    self._tolerance_violated_during_insertion = True
                reward += depth_progress * 5.0
                reward -= xy_offset * 3.0
                reward -= tilt * 1.0

        insertion_success = (
            self._insertion_started
            and depth_progress >= INSERTION_DEPTH
            and not self._tolerance_violated_during_insertion
        )
        if insertion_success:
            reward += 30.0  # larger than Phase 2's place bonus - this is the harder, final task

        terminated = bool(insertion_success)

        obs = self._get_obs()
        info["success"] = insertion_success
        info["insertion_success"] = insertion_success
        info["xy_offset_from_hole"] = xy_offset
        info["peg_tilt_rad"] = tilt
        info["insertion_depth_progress"] = depth_progress
        info["board_collision"] = board_collision
        info["tolerance_violated_during_insertion"] = self._tolerance_violated_during_insertion

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = PegInsertEnv(render_mode=None)
    obs, info = env.reset()
    print("Observation shape:", obs.shape, "| Action shape:", env.action_space.shape)
    print("Hole xy this episode:", env._hole_xy)
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
