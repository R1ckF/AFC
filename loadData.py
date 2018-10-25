import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd

prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']


def loadData(filename):
    with open(os.path.join("results2",filename), 'rb') as f:
        return pickle.load(f)


def sort_lists(R,T,E):
    order = np.argsort(T)
    R = [R[i] for i in order]
    T = [T[i] for i in order]
    E = [E[i] for i in order]

    return R,T,E, np.arange(len(R))

def prepPlot(name):

    Rewards = []
    Time = []
    ElTime = []
    for i in range(9):
        r, t, et = loadData(name+str(i))
        Rewards.append(r)
        Time.append(t)
        ElTime.append(et)
    Rewards, Time, ElTime = np.hstack(Rewards),np.hstack(Time), np.hstack(ElTime)
    RewardsT,TimeT,ElTimeT,nEpisodesT = sort_lists(Rewards, Time, ElTime)
    r = pd.Series(RewardsT)
    movingAv = r.rolling(window).mean()
    movingStd = r.rolling(window).std()
    return avReward, stdReward, TimeT, ElTimeT, nEpisodesT






plt.figure()
plt.plot(TimeT, movingAv,color=COLORS[0])
plt.fill_between(TimeT,movingAv-1*movingStd,movingAv+1*movingStd,color=COLORS[0],alpha=0.2)
plt.xlabel('Timestep')
plt.ylabel('Reward')
# plt.show()

# plt.figure()
# plt.plot(np.arange(len(allEpR)),allEpR)
# plt.xlabel('Episode')
# plt.ylabel('Reward')

plt.figure()
plt.scatter(TimeT,RewardsT)
plt.xlabel('Timestep')
plt.ylabel('Reward')
#
# # plt.figure()
# # plt.plot(ElapsedTime,allEpR)
# # plt.xlabel('Time [s]')
# # plt.ylabel('Reward')
#
plt.show()
