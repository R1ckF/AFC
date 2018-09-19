import gym
import numpy as np
import cv2
from gym import spaces
from collections import deque

class adjustFrame(gym.ObservationWrapper):

    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        Also removes top rows as they do not contain info for the game

        """

        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[25:,:]
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :,None]

    def close(self):
        return self.env.close()

class stackFrames(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.
        This is to let the NN able to extract information about velocity and acceleration.
        Also normalizes observation for faster converges
        """

        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=self.k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0], shp[1], shp[2] * self.k), dtype=np.float32)

    def reset(self):
        ob = self.env.reset().astype(np.float32)/np.float32(255.0)
        for _ in range(self.k):
            self.frames.append(ob)
        return np.concatenate(self.frames,axis=2)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob.astype(np.float32)/np.float32(255.0))
        return np.concatenate(self.frames,axis=2), reward, done, info

    def close(self):
        return self.env.close()
