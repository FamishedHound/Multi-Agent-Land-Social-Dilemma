import pickle

import matplotlib.pyplot as plt


#
# # #ToDo Single agent analysis
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\incentive_tracker_new.pkl', 'rb') as f:
# #     a = pickle.load(f)
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\agents_reward_new.pkl', 'rb') as f:
# #     b = pickle.load(f)
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\average_new.pkl', 'rb') as f:
# #     c = pickle.load(f)
# #
# # plt.plot(a)
# # plt.show()
# # plt.plot(b)
# # plt.show()
# # plt.axhline(y = 0.15, color = 'r', linestyle = '-')
# # plt.plot(c[300:])
# # plt.show()
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\incentive_tracker_new_baseline.pkl', 'rb') as f:
# #     a = pickle.load(f)
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\agents_reward_new_baseline.pkl', 'rb') as f:
# #     b = pickle.load(f)
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\average_new_baseline.pkl', 'rb') as f:
# #     c = pickle.load(f)
# #
# # plt.plot(a, color = 'pink')
# # plt.show()
# # plt.plot(b, color = 'pink')
# # plt.show()
# # plt.axhline(y = 0.15, color = 'r', linestyle = '-')
# # plt.plot(c, color = 'pink')
# # plt.show()
# #ToDo Multi-Agent analysis
# target = [0.35, 0.65, 0.25, 0.2]
# #ToDo multiple agents analysis
# # with open('/multiagent-particle-envs/important_pickles/multiple_agents_incentive_tracker_new_baseline.pkl', 'rb') as f:
# #     a = pickle.load(f)
# # with open('/multiagent-particle-envs/important_pickles/multiple_agents_average_new_baseline.pkl', 'rb') as f:
# #     b = pickle.load(f)
# # with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\mixed_motive_distance_from_target45.pkl', 'rb') as f:
# #     c = pickle.load(f)
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\spending45.pkl', 'rb') as f:
#     d = pickle.load(f)
# # #print(c)
# # #new_list = [sum(x) for x in c]
# # # plt.plot(c, color = 'red')
# # # plt.show()
# # # a = a[:2000]
# # # b = b[:2000]
# # # c = c[:2000]
# # print(8%4)
# # agent_0_averages =[]
# # agent_1_averages =[]
# # agent_2_averages =[]
# # agent_3_averages =[]
# #
# # agent_0_averages_b =[]
# # agent_1_averages_b =[]
# # agent_2_averages_b =[]
# # agent_3_averages_b =[]
# # import numpy as np
# #
# # for average in d:
# #     agent_0_averages_b.append(average[0])
# #
# #     agent_1_averages_b.append(average[1])
# #
# #     agent_2_averages_b.append(average[2])
# #
# #     agent_3_averages_b.append(average[3])
# # for average in c:
# #     agent_0_averages.append(average[0])
# #
# #     agent_1_averages.append(average[1])
# #
# #     agent_2_averages.append(average[2])
# #
# #     agent_3_averages.append(average[3])
# #
#
# #print(f"agent 0 has following variance : {np.var(1)} and std {}")
#
# #all_of_them  = [agent_0_averages,agent_1_averages,agent_2_averages,agent_3_averages]
# # for i in range(4):
# #     agent = [x[i] for x in a]
# #     plt.axhline(y=target[i], color='r', linestyle='-')
# #     plt.plot(agent, color = 'orange')
# #     plt.show()
# #
# # for i in range(4):
# #     agent = [x[i] for x in b]
# #     plt.axhline(y=target[i], color='r', linestyle='-')
# #     plt.plot(agent, color = 'green')
# #     plt.show()
#
# def flatten(t):
#     return [item for sublist in t for item in sublist]
#
# # ax = plt.gca()
# # #ax.set_ylim([-1, 1])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_0_averages, color = 'green')
# # plt.savefig('mm_agent_0.png')
# # plt.show()
# # ax = plt.gca()
# # ax.set_ylim([-0.3, 0.3])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_0_averages_b, color = 'black')
# # plt.savefig('mm_agent_0_b.png')
# # plt.show()
# # ax = plt.gca()
# # #ax.set_ylim([-0.3, 0.3])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_1_averages, color = 'green')
# # plt.savefig('mm_agent_1.png')
# # plt.show()
# # ax = plt.gca()
# # #ax.set_ylim([-0.3, 0.3])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_1_averages_b, color = 'black')
# # plt.savefig('mm_agent_1_b.png')
# # plt.show()
# # ax = plt.gca()
# # #ax.set_ylim([-0.3, 0.3])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_2_averages, color = 'green')
# # plt.savefig('mm_agent_2.png')
# # plt.show()
# # ax = plt.gca()
# # #ax.set_ylim([-0.3, 0.3])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_2_averages_b, color = 'black')
# # plt.savefig('mm_agent_2_b.png')
# # plt.show()
# # ax = plt.gca()
# # #ax.set_ylim([-0.3, 0.3])
# #
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_3_averages, color = 'green')
# # plt.savefig('mm_agent_3.png')
# # plt.show()
# # ax = plt.gca()
# # #ax.set_ylim([-0.3, 0.3])
# # plt.axhline(y=0, color='r', linestyle='-')
# # plt.plot(agent_3_averages_b, color = 'black')
# # plt.savefig('mm_agent_3_b.png')
# # plt.show()
# from scipy.spatial import distance_matrix
#
# import numpy as np
#
# list_a = np.array([[3,2,1,2]])
# list_b = np.array([[3,1,1,1]])
#
# def run_euc(list_a,list_b):
#     return np.array([[ np.linalg.norm(i-j) for j in list_b] for i in list_a])
# print(distance_matrix(list_a, list_b)[0][0])
# print(run_euc(list_a, list_b))
# def measure_sum_of_distances_between_matrices(a,b):
#     distances = []
#     for x,y in zip(a,b):
#         for z,c in zip(x,y):
#             distances.append(abs(z-c))
#
#     return sum(distances)
# with open('C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\spending45.pkl', 'rb') as f:
#     c = pickle.load(f)
#
# # print(c)
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
# #plt.axhline(y=0, color='r', linestyle='-')
# ax = plt.gca()
# #ax.set_ylim([-2, 4])
# plt.plot(agent_0_baseline, color = 'black')
# plt.savefig('s_agent_0_b.png')
# plt.show()
#
# #plt.axhline(y=0, color='r', linestyle='-')
# ax = plt.gca()
# #ax.set_ylim([-2, 4])
# plt.plot(agent_1_baseline, color = 'black')
# plt.savefig('s_agent_1_b.png')
#
# plt.show()
# ax = plt.gca()
# #ax.set_ylim([-2, 4])
# #plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_2_baseline, color = 'black')
# plt.savefig('s_agent_2_b.png')
# plt.show()
# ax = plt.gca()
# #ax.set_ylim([-2, 4])
# #plt.axhline(y=0, color='r', linestyle='-')
# plt.plot(agent_3_baseline, color = 'black')
# plt.savefig('s_agent_3_b.png')
# plt.show()

def visualise(path_to_oa, path_to_baseline, mode, name_of_study):
    with open(path_to_oa,
              'rb') as f:
        c = pickle.load(f)
    with open(path_to_baseline, 'rb') as f:
        d = pickle.load(f)
    agent_0_averages = []
    agent_1_averages = []
    agent_2_averages = []
    agent_3_averages = []

    agent_0_averages_b = []
    agent_1_averages_b = []
    agent_2_averages_b = []
    agent_3_averages_b = []
    if mode == 'spending' or mode == 'reward':
        for average in range(4, len(d)):
            if average % 4 == 0:
                agent_0_averages_b.append(d[average])
            if average % 4 == 1:
                agent_1_averages_b.append(d[average])
            if average % 4 == 2:
                agent_2_averages_b.append(d[average])
            if average % 4 == 3:
                agent_3_averages_b.append(d[average])
        for average in range(4, len(c)):
            if average % 4 == 0:
                agent_0_averages.append(c[average])
            if average % 4 == 1:
                agent_1_averages.append(c[average])
            if average % 4 == 2:
                agent_2_averages.append(c[average])
            if average % 4 == 3:
                agent_3_averages.append(c[average])
    if mode == 'distance':
        for average in d:
            agent_0_averages_b.append(average[0])

            agent_1_averages_b.append(average[1])

            agent_2_averages_b.append(average[2])

            agent_3_averages_b.append(average[3])
        for average in c:
            agent_0_averages.append(average[0])

            agent_1_averages.append(average[1])

            agent_2_averages.append(average[2])

            agent_3_averages.append(average[3])
    ax = plt.gca()
    if mode == 'distance':
        # left, right = plt.xlim()

        ax = plt.gca()
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(agent_0_averages_b)])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(agent_0_averages, color='green')
        plt.plot(agent_0_averages_b, color='black')
        plt.savefig(f'to_upload/{name_of_study}_agent_0.png', bbox_inches="tight")
        plt.show()

        ax = plt.gca()
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(agent_0_averages_b)])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(agent_1_averages, color='green')
        plt.plot(agent_1_averages_b, color='black')
        plt.savefig(f'to_upload/{name_of_study}_agent_1.png', bbox_inches="tight")
        plt.show()

        ax = plt.gca()
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(agent_0_averages_b)])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(agent_2_averages, color='green')
        plt.plot(agent_2_averages_b, color='black')
        plt.savefig(f'to_upload/{name_of_study}_agent_2.png', bbox_inches="tight")
        plt.show()
        ax = plt.gca()
        # ax.set_ylim([-0.3, 0.3])

        plt.axhline(y=0, color='r', linestyle='-')
        ax = plt.gca()
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(agent_0_averages_b)])
        plt.plot(agent_3_averages, color='green')
        plt.plot(agent_3_averages_b, color='black')
        plt.savefig(f'to_upload/{name_of_study}_agent_3.png', bbox_inches="tight")
        plt.show()
    if mode == 'spending' or mode == "reward":
        import numpy as np

        # plt.axhline(y=0, color='r', linestyle='-')
        ax = plt.gca()
        # ax.set_ylim([-2, 4])
        if mode == 'reward':
            plt.axhline(np.mean(np.array(agent_0_averages)), color='red', label="mean of OA agent", linestyle="--")
            plt.axhline(np.mean(np.array(agent_0_averages_b)), color='blue', label="mean of baseline agent",
                        linestyle="--")
        elif mode == 'spending':
            plt.axhline(np.mean(np.array(agent_0_averages)), color='red', label="mean of OA agent", linestyle="--")
        ax.set_xlim([0, len(agent_0_averages_b)])
        plt.plot(agent_0_averages, color='green')
        plt.plot(agent_0_averages_b, color='black')
        plt.savefig(f'to_upload/{name_of_study}_agent_0_b.png', bbox_inches="tight")
        plt.legend(loc='center', bbox_to_anchor=(1, 0))
        plt.show()
        if mode == 'reward':

            plt.axhline(np.mean(np.array(agent_1_averages)), color='red', label="mean of OA agent", linestyle="--")
            plt.axhline(np.mean(np.array(agent_1_averages_b)), color='blue', label="mean of baseline agent",
                        linestyle="--")
        elif mode == 'spending':
            plt.axhline(np.mean(np.array(agent_1_averages)), color='red', label="mean of OA agent", linestyle="--")
        # plt.axhline(y=0, color='r', linestyle='-')
        ax = plt.gca()
        # ax.set_ylim([-2, 4])
        ax.set_xlim([0, len(agent_0_averages_b)])
        plt.plot(agent_1_averages, color='green')
        plt.plot(agent_1_averages_b, color='black')
        plt.savefig(f'to_upload/{name_of_study}_agent_1_b.png', bbox_inches="tight")
        plt.legend(loc='center', bbox_to_anchor=(1, 0))
        plt.show()

        ax = plt.gca()
        if mode == 'reward':
            plt.axhline(np.mean(np.array(agent_2_averages)), color='red', label="mean of OA agent", linestyle="--")
            plt.axhline(np.mean(np.array(agent_2_averages_b)), color='blue', label="mean of baseline agent",
                        linestyle="--")
        elif mode == 'spending':
            plt.axhline(np.mean(np.array(agent_2_averages)), color='red', label="mean of OA agent", linestyle="--")
        ax.set_xlim([0, len(agent_0_averages_b)])
        # ax.set_ylim([-2, 4])
        # plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(agent_2_averages, color='green')
        plt.plot(agent_2_averages_b, color='black')

        plt.legend(loc='center', bbox_to_anchor=(1, 0))
        plt.savefig(f'to_upload/{name_of_study}_agent_2_b.png', bbox_inches="tight")
        plt.show()
        ax = plt.gca()
        ax.set_xlim([0, len(agent_0_averages_b)])
        # ax.set_ylim([-2, 4])
        # plt.axhline(y=0, color='r', linestyle='-')
        if mode == 'reward':
            plt.axhline(np.mean(np.array(agent_3_averages)), color='red', label="mean of OA agent", linestyle="--")
            plt.axhline(np.mean(np.array(agent_3_averages_b)), color='blue', label="mean of baseline agent",
                        linestyle="--")
        elif mode == 'spending':
            plt.axhline(np.mean(np.array(agent_3_averages)), color='red', label="mean of OA agent", linestyle="--")
        plt.plot(agent_2_averages, color='green')
        plt.plot(agent_3_averages_b, color='black')

        plt.legend(loc='center', bbox_to_anchor=(1, 0))
        plt.savefig(f'to_upload/{name_of_study}_agent_3_b.png', bbox_inches="tight")
        plt.show()


base_path = "C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\"


# visualise(base_path+"mixed_motive_distance_from_target45.pkl",base_path+"mixed_motive_distance_from_target_baseline45.pkl","distance","distance_from_taget")
# visualise(base_path+"all_env_distance_from_target45.pkl",base_path+"all_env_distance_from_target_baseline45.pkl","distance","greedy_distance")
# visualise(base_path+"all_greedy_incetives_given45.pkl",base_path+"all_greedy_spending45.pkl","spending","greedy_spending")
# visualise(base_path+"all_greedy_personal_reward45.pkl",base_path+"all_greedy_baseline_reward_without_incentive45.pkl","reward","greedy_reward")
# studies = ['all_greedy_','all_env_','fake_planned_','mixed_motive_']
# studies = ['fake_planned_50_50_']
# metrics = [('distance_from_target45.pkl','distance_from_target_baseline45.pkl','distance'),('incetives_given45.pkl','spending45.pkl','spending'),('personal_reward45.pkl','baseline_reward_without_incentive45.pkl','reward')]
# for study in studies:
#     for metric in metrics:
#         visualise(base_path + f"{study}{metric[0]}",
#                   base_path + f"{study}{metric[1]}", metric[2], f'{study}_{metric[2]}')
def get_the_data(path_to_oa, path_to_baseline, mode, name_of_study):
    import numpy as np
    with open(path_to_oa,
              'rb') as f:
        c = pickle.load(f)
    with open(path_to_baseline, 'rb') as f:
        d = pickle.load(f)
    agent_0_averages = []
    agent_1_averages = []
    agent_2_averages = []
    agent_3_averages = []

    agent_0_averages_b = []
    agent_1_averages_b = []
    agent_2_averages_b = []
    agent_3_averages_b = []
    if mode == 'spending' or mode == 'reward':
        for average in range(4, len(d)):
            if average % 4 == 0:
                agent_0_averages_b.append(d[average])
            if average % 4 == 1:
                agent_1_averages_b.append(d[average])
            if average % 4 == 2:
                agent_2_averages_b.append(d[average])
            if average % 4 == 3:
                agent_3_averages_b.append(d[average])
        for average in range(4, len(c)):
            if average % 4 == 0:
                agent_0_averages.append(c[average])
            if average % 4 == 1:
                agent_1_averages.append(c[average])
            if average % 4 == 2:
                agent_2_averages.append(c[average])
            if average % 4 == 3:
                agent_3_averages.append(c[average])
        averages_oa = (np.array(agent_0_averages).mean() + np.array(agent_1_averages).mean() + np.array(
            agent_2_averages).mean() + np.array(agent_3_averages).mean()) / 4

        average_baseline = (np.array(agent_0_averages_b).mean() + np.array(agent_1_averages_b).mean() + np.array(
            agent_2_averages_b).mean() + np.array(agent_3_averages_b).mean()) / 4
    if mode == 'distance':
        for average in d:
            agent_0_averages_b.append(average[0])

            agent_1_averages_b.append(average[1])

            agent_2_averages_b.append(average[2])

            agent_3_averages_b.append(average[3])
        for average in c:
            agent_0_averages.append(average[0])

            agent_1_averages.append(average[1])

            agent_2_averages.append(average[2])

            agent_3_averages.append(average[3])
        averages_oa = (np.array(agent_0_averages).mean() + np.array(agent_1_averages).mean() + np.array(
            agent_2_averages).mean() + np.array(agent_3_averages).mean()) / 4
        average_baseline = (np.array(agent_0_averages_b).mean() + np.array(agent_1_averages_b).mean() + np.array(
            agent_2_averages_b).mean() + np.array(agent_3_averages_b).mean()) / 4
    return averages_oa,average_baseline
import pandas as pd
path = "C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\important_pickles\\fake_planned_{}_{}"
#,('incetives_given45.pkl','spending45.pkl','spending'),('personal_reward45.pkl','baseline_reward_without_incentive45.pkl','reward')
metrics = [('distance_from_target45.pkl','distance_from_target_baseline45.pkl','distance'),('incetives_given45.pkl','spending45.pkl','spending'),('personal_reward45.pkl','baseline_reward_without_incentive45.pkl','reward')]
df_dict = {}
for index in ["10_90","20_80","30_70","40_60","50_50","60_40","70_30","80_20","90_10"]: #ToDo lacking 20_80
    if "index" not in df_dict:
        df_dict["index"] = []
    for metric in metrics:
        if metric[2] not in df_dict:
            df_dict[metric[2]] = []
            df_dict[metric[2]+'_b'] = []
        oa,baseline_agent = get_the_data(path.format(index,metric[0]),
                           path.format(index,metric[1]), metric[2], f'{index}_{metric[2]}')
        df_dict[metric[2]].append(oa)
        df_dict[metric[2] + '_b'].append(baseline_agent)
    df_dict["index"].append(index)

print(df_dict)
df = pd.DataFrame.from_dict(df_dict).set_index("index")
print(df.head(20))
plt.plot(df["distance"],color="green")
plt.plot(df["distance_b"],color="black")
plt.show()

plt.plot(df["spending"],color="green")
plt.plot(df["spending_b"],color="black")
plt.show()

plt.plot(df["reward"],color="green")
plt.plot(df["reward_b"],color="black")
plt.show()