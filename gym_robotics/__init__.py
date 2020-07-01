from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
      suffix = 'Dense' if reward_type =='dense' else ''
      kwargs = {
            'reward_type': reward_type
      }

      register(
            id='Test{}-v1'.format(suffix),
            entry_point='gym_robotics.envs:FetchPickAndPlaceEnv',
            kwargs=kwargs,
            max_episode_steps=50,
      )

      register(
            id='TwoBlocks{}-v1'.format(suffix),
            entry_point='gym_robotics.envs:FetchTwoBlocksEnv',
            kwargs=kwargs,
            max_episode_steps=50,
      )
