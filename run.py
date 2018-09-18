import gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import cv2
## parsing function for easy running
def parse_args():
        parser = argparse.ArgumentParser(description='Plot results of Simulations')
        parser.add_argument('--env', default="SpaceInvaders-v0")
        parser.add_argument('--results_path', default="results/")
        parser.add_argument('--play', action='store_true')
        args = parser.parse_args()
        return args

args=parse_args()

##create environement
env = gym.make(args.env)
env = gym.wrappers.Monitor(env, args.results_path, force=True)
stacked = deque([],maxlen=3)
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)
# print(env.action_space)
# print(env.unwrapped.get_action_meanings())
obs = env.reset()
plt.imshow(obs)
plt.show()
obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
stacked.append(obs)

stacked.append
print(obs)
print(obs.shape)
plt.imshow(obs)
plt.show()
