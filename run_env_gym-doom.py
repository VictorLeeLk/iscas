import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-doom')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/DoomBasic-v0')