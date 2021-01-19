import numpy as np
from gym import utils
from . import mujoco_env
import math

class AntWallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'antwall.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        action_scaled = action * self.action_range + self.action_center
        #print(action, action_scaled)
        self.do_simulation(action_scaled, self.frame_skip)
        next_obs = self._get_obs()
        qpos = self.get_body_com("torso")[:2]
        goal = [15.0, 0.0]
        reward = -np.linalg.norm(goal - qpos)
        return next_obs, reward, False, {}

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        #print('qpos',self.sim.data.qpos.copy())
        #print('qvel',self.sim.data.qvel.copy())
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
