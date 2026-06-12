import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os


class MyRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Get the directory where this file is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, "my_robot.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        # RL action range stays -1 to +1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Observation = joint positions + joint velocities + tip position + target
        obs_size = self.model.nq + self.model.nv + 3 + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Target spawning region: 0.6x0.6 area centered at (0.55, 0, 0.1)
        # X range: 0.25 to 0.85, Y range: -0.3 to 0.3, Z fixed at 0.1
        self.target_x_min = 0.25
        self.target_x_max = 0.85
        self.target_y_min = -0.3
        self.target_y_max = 0.3
        self.target_z = 0.1  # Fixed height
        
        # Initialize target at a random position
        self.target = self._sample_target_position()

        # Episode limit
        self.max_steps = 300
        self.current_step = 0

        # Make control stronger for better reaching
        # Action from PPO is between -1 and 1,
        # but now 100% is sent to the motors (stronger control).
        self.action_scale = 1.0

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self._get_tip_position(),
            self.target,
        ])

        return obs.astype(np.float32)

    def _sample_target_position(self):
        """Sample a random target position within the reachable workspace."""
        x = self.np_random.uniform(self.target_x_min, self.target_x_max)
        y = self.np_random.uniform(self.target_y_min, self.target_y_max)
        z = self.target_z
        return np.array([x, y, z], dtype=np.float32)

    def _get_tip_position(self):
        tip_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tip",
        )

        return self.data.xpos[tip_id].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)

        # Get the base body and its joint addresses
        base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base",
        )
        base_jntadr = self.model.body_jntadr[base_body_id]
        
        # Initialize robot joints with random positions across full 180-degree range
        # Get the joint IDs for the 6 robot joints
        joint_ids = []
        for i in range(1, 7):  # joint1 through joint6
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"joint{i}",
            )
            joint_ids.append(joint_id)
        
        # Set random angles for each joint
        for i, joint_id in enumerate(joint_ids):
            jntadr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[jntadr] = self.np_random.uniform(
                low=-np.pi/2,
                high=np.pi/2,
            )

        # Sample new target position for this episode
        self.target = self._sample_target_position()
        
        # Set target body position in MuJoCo (first 3 elements of qpos for the target free joint)
        target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )
        target_qpos_adr = self.model.body_jntadr[target_body_id]
        # Free joint has 7 DOF: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[target_qpos_adr:target_qpos_adr+3] = self.target
        # Set default orientation (identity quaternion) for the target
        self.data.qpos[target_qpos_adr+3:target_qpos_adr+7] = [1, 0, 0, 0]

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Forward kinematics to propagate changes
        mujoco.mj_forward(self.model, self.data)

        info = {
            "target": self.target.copy(),
            "tip_position": self._get_tip_position(),
        }

        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Scale action before sending to MuJoCo
        scaled_action = action * self.action_scale

        self.data.ctrl[:] = scaled_action

        # Run several MuJoCo physics steps per one RL step
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        tip_pos = self._get_tip_position()
        distance = np.linalg.norm(tip_pos - self.target)

        # Main reward: negative distance (normalized)
        # Scale distance to [-1, 0] range where -1 is max distance
        max_distance = 2.0
        normalized_distance = min(distance / max_distance, 1.0)
        reward = -normalized_distance

        # Small penalty for energy use (encourage efficiency but not blocking movement)
        reward -= 0.01 * float(np.sum(np.square(action)))

        # Success bonus - generous to encourage reaching
        if distance < 0.05:
            reward += 5.0
        elif distance < 0.1:
            reward += 3.0
        elif distance < 0.2:
            reward += 1.0

        terminated = bool(distance < 0.05)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "distance": float(distance),
            "target": self.target.copy(),
            "tip_position": tip_pos.copy(),
            "raw_action": action.copy(),
            "scaled_action": scaled_action.copy(),
            "is_success": bool(distance < 0.05),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None