import pickle

import matplotlib.pyplot as plt


with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\pytorch-maddpg\\before_incentive_reward.pkl', 'rb') as f:
    a = pickle.load(f)
with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\pytorch-maddpg\\incentive_reward.pkl', 'rb') as f:
    b = pickle.load(f)
with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\pytorch-maddpg\\difference_in_reward.pkl', 'rb') as f:
    c = pickle.load(f)

plt.plot(a)
plt.show()
plt.plot(b)
plt.show()
plt.plot(c)
plt.show()
print(25 % 5)