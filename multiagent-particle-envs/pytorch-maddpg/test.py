import pickle

import matplotlib.pyplot as plt
#
# #ToDo Single agent analysis
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\incentive_tracker_new.pkl', 'rb') as f:
#     a = pickle.load(f)
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\agents_reward_new.pkl', 'rb') as f:
#     b = pickle.load(f)
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\average_new.pkl', 'rb') as f:
#     c = pickle.load(f)
#
# plt.plot(a)
# plt.show()
# plt.plot(b)
# plt.show()
# plt.axhline(y = 0.15, color = 'r', linestyle = '-')
# plt.plot(c[300:])
# plt.show()
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\incentive_tracker_new_baseline.pkl', 'rb') as f:
#     a = pickle.load(f)
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\agents_reward_new_baseline.pkl', 'rb') as f:
#     b = pickle.load(f)
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\average_new_baseline.pkl', 'rb') as f:
#     c = pickle.load(f)
#
# plt.plot(a, color = 'pink')
# plt.show()
# plt.plot(b, color = 'pink')
# plt.show()
# plt.axhline(y = 0.15, color = 'r', linestyle = '-')
# plt.plot(c, color = 'pink')
# plt.show()
#ToDo Multi-Agent analysis
target = [0.15,0.3,0.1,0.5]
#ToDo multiple agents analysis
# with open('/multiagent-particle-envs/important_pickles/multiple_agents_incentive_tracker_new_baseline.pkl', 'rb') as f:
#     a = pickle.load(f)
# with open('/multiagent-particle-envs/important_pickles/multiple_agents_average_new_baseline.pkl', 'rb') as f:
#     b = pickle.load(f)
with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\incetives_given.pkl', 'rb') as f:
    c = pickle.load(f)
print(c)
new_list = [sum(x) for x in c]
plt.plot(new_list, color = 'red')
plt.show()
# a = a[:2000]
# b = b[:2000]
# c = c[:2000]
print(8%4)
agent_0_averages =[]
agent_1_averages =[]
agent_2_averages =[]
agent_3_averages =[]


for average in c:

    agent_0_averages.append(average[0])

    agent_1_averages.append(average[1])

    agent_2_averages.append(average[2])

    agent_3_averages.append(average[3])
print(agent_0_averages)
all_of_them  = [agent_0_averages,agent_1_averages,agent_2_averages,agent_3_averages]
# for i in range(4):
#     agent = [x[i] for x in a]
#     plt.axhline(y=target[i], color='r', linestyle='-')
#     plt.plot(agent, color = 'orange')
#     plt.show()
#
# for i in range(4):
#     agent = [x[i] for x in b]
#     plt.axhline(y=target[i], color='r', linestyle='-')
#     plt.plot(agent, color = 'green')
#     plt.show()

# def flatten(t):
#     return [item for sublist in t for item in sublist]
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_0_averages, color = 'green')
# plt.show()
#
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_1_averages, color = 'green')
# plt.show()
#
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_2_averages, color = 'green')
# plt.show()
#
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_3_averages, color = 'green')
# plt.show()
#
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\distance_from_target_baseline.pkl', 'rb') as f:
#     c = pickle.load(f)
# print(c)
# agent_0_baseline =[c[0]]
# agent_1_baseline =[c[1]]
# agent_2_baseline =[c[2]]
# agent_3_baseline =[c[3]]
# for average in range(4,len(c)):
#     if average%4==0:
#         agent_0_baseline.append(c[average])
#     if average % 4 == 1:
#         agent_1_baseline.append(c[average])
#     if average % 4 == 2:
#         agent_2_baseline.append(c[average])
#     if average % 4 == 3:
#         agent_3_baseline.append(c[average])
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_0_baseline, color = 'green')
# plt.show()
#
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_1_baseline, color = 'green')
# plt.show()
#
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_2_baseline, color = 'green')
# plt.show()
#
# plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_3_baseline, color = 'green')
# plt.show()