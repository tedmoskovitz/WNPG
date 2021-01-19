from gym.envs.registration import register

register(
		id = 'point-v0',
		entry_point='simpleenvs.point:PointEnv',
		max_episode_steps=50,
		)


register(
		id = 'antwall-v0',
		entry_point='simpleenvs.antwall:AntWallEnv',
		max_episode_steps=400,
		)