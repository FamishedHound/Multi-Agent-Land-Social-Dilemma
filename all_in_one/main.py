import random
import warnings

from MADDPG import MADDPG
import numpy as np
import torch as th
from goverment_agent import goverment_agent as meta_agent
from gym_pollinator_game import gymDriver
from params import scale_reward


def measure_sum_of_distances_between_matrices(a, b):
    distances = []
    for x, y in zip(a, b):
        for z, c in zip(x, y):
            distances.append(abs(round(z.item(),1) - round(c.item(),1)))

    return sum(distances)

def make_random_action(all_agents):
    action = []
    for agent in all_agents:
        agent_actions = [th.tensor(random.choice([random.uniform(0, 102)]) / 100) for _ in
                         range(len(agent.land_cells_owned))]

        action.append(agent_actions)
    return action


e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = gymDriver()

reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.seed(1234)

n_states = 213
n_actions = 1
capacity = 9000 #ToDo was 500
batch_size = 16
n_episode = 500
max_steps = 100000000
episodes_before_train = 10
epsilon = 1
win = None
param = None
obs = world.reset()
# initialization of the objeects is after reset that's why it's here
dims = np.array(obs)
n_agents = world.n_agents
buffer_action = None
buffer_counter = 0
cum_reward = []
import matplotlib.pyplot as plt

(obs, global_state) = world.reset()
worlds_all_agents = world.agent_processor.all_agents
maddpg = MADDPG(world.n_agents, 12, n_actions, batch_size, capacity,
                episodes_before_train, worlds_all_agents,False)
maddpg_fake = MADDPG(world.n_agents, 12, n_actions, batch_size, capacity,
                episodes_before_train, worlds_all_agents,False)
FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor


meta = meta_agent(None, None, worlds_all_agents)
counters = 0
incentive = [0,0,0,0]
warnings.filterwarnings("ignore")
flag_for_incentive = False
flag_for_real_action = False
incentive_reward = []
before_incentive_reward = []
difference_in_reward = []
reward =[]
incentive_tracker=[]
average = []
baseline = []
baseline_spending = []
reward_without_incentive_baseline = []

fake_real_difference = []
def handle_exploration():
    global epsilon
    if maddpg.episode_done >= batch_size:
        # if epsilon > 0.6 :
        #     epsilon -= 1e-3
        # elif epsilon <  0.6 and epsilon > 0.3:
        #     epsilon -= 1e-4
        # el
        #epsilon -= 1e-4  # 0.0001
        #ToDo static epsilon going 0
        if round(epsilon * 1000) % 211 == 0 and epsilon == 0.3:
            plt.plot(cum_reward)
            plt.show()
        if epsilon > 0.4:
            epsilon -= 1e-4 # 0.0001
        else:
            epsilon = 0.4


def cirriculum_learning(worlds_all_agents,action,epsilon,counters):

    if epsilon <= 0.8 and counters <= 100:
        incentive = meta.distribute_incetive()
    if counters >= 100 and counters < 1200:
        incentive = meta.distribute_incentive_2()
    if counters >= 1200 and counters < 1800:
        incentive = []
        for i, a in enumerate(worlds_all_agents):
            average_action = sum([x.item() for x in action[i]]) / len(action[i])
            print(f" Average :  {average_action} !! ")
            incentive.append(1 - (abs(meta.target[i] - average_action)))
            baseline.append(meta.target[i] - average_action)
    if counters >= 1800 and counters < 2000:
        incentive = meta.distribute_incentive_4()
    else:
        incentive=[0,0,0,0]
    return incentive

if __name__ == "__main__":
    for i_episode in range(n_episode):

        # obs = np.stack(obs)
        # if isinstance(obs, np.ndarray):
        #     obs = th.from_numpy(obs).float()
        total_reward = 0.0
        rr = np.zeros((n_agents,))
        for t in range(max_steps):
            # render every 100 episodes to speed up training
            # if i_episode % 100 == 0 and e_render:
            world.clockobject.tick(99)
            # world.render()

            # obs = obs.type(FloatTensor)
            randy_random = random.uniform(0, 1)

            # if epsilon<0.9 and epsilon>0.3:
            #     incentive = meta.distribute_incetive()
            # if epsilon<0.5 and counters<=19700:
            #     incentive = meta.distribute_incentive_4()
            #incentive = meta.optimise_incentives(obs, worlds_all_agents, maddpg.actors, maddpg.critics)
            #incentive = meta.optimise_incentives(obs, worlds_all_agents, maddpg.actors, maddpg.critics)
            #incentive = [0, 0, 0, 0]



            # print(f" action for the game {agents_actions}")
            # action = th.from_numpy(action_np)
            # obs_, reward, done, _ = world.step(action, randy_random_2)
            print("Epsilon is {}".format(epsilon))
            if epsilon <= 0.4:
                counters += 1
                # maddpg.save_weights_of_networks("before_curriculum")
                print("The COUNTER IS {}".format(counters))




            if counters>=3300:
                import pickle

                with open('important_pickles/fake_planned_20_80_personal_reward45.pkl', 'wb') as f:
                    pickle.dump(difference_in_reward, f)
                with open('important_pickles/fake_planned_20_80_incetives_given45.pkl', 'wb') as f:
                    pickle.dump(incentive_tracker, f)
                with open('important_pickles/fake_planned_20_80_distance_from_target45.pkl', 'wb') as f:
                    pickle.dump(average, f)
                with open('important_pickles/fake_planned_20_80_distance_from_target_baseline45.pkl', 'wb') as f:
                    pickle.dump(baseline, f)
                with open('important_pickles/fake_planned_20_80_spending45.pkl', 'wb') as f:
                    pickle.dump(baseline_spending, f)
                with open('important_pickles/fake_planned_20_80_baseline_reward_without_incentive45.pkl', 'wb') as f:
                    pickle.dump(reward_without_incentive_baseline, f)
                exit(1)


            if counters>=2100:
                import pickle
                with open('important_pickles/distance_real_fake.pkl', 'wb') as f:
                    pickle.dump(fake_real_difference, f)
                #incentive,new_ob =meta.optimise_incentives(obs,worlds_all_agents,maddpg.actors,maddpg.critics) #ToDo now Fake one is optimised
                incentive, new_ob = meta.optimise_incentives(obs, worlds_all_agents, maddpg_fake.actors, maddpg_fake.critics)
                obs = new_ob
            print(f"Optimiser gave the following incentive : {incentive}")
            if epsilon > randy_random:
                action = maddpg.select_action(obs, worlds_all_agents)
                action_fake = maddpg_fake.select_action(obs, worlds_all_agents) #ToDo fake maddpg
                flag_for_real_action = False
                agents_actions = ''.join(f'''agent {i} made actions {a} \n ''' for i, a in enumerate(action))
                agents_actions_fake = ''.join(f''' FAKE agent {i} made actions {a} \n ''' for i, a in enumerate(action_fake)) #ToDo Fake agent
                print(agents_actions)
                print(agents_actions_fake) #ToDo Fake agent
                action = make_random_action(worlds_all_agents)
                distance = measure_sum_of_distances_between_matrices(action,action_fake)
                fake_real_difference.append(distance)
            else:
                # maddpg.batch_size= 32
                flag_for_real_action = True
                action = maddpg.select_action(obs, worlds_all_agents)
                action_fake = maddpg_fake.select_action(obs, worlds_all_agents)  # ToDo fake maddpg
                distance = measure_sum_of_distances_between_matrices(action, action_fake)
                fake_real_difference.append(distance)

            print(f"DISTANCE BETWEEN FAKE AND REAL IS {distance}")
            if epsilon <= 0.8 and counters <= 100:
                incentive = meta.distribute_incetive()
            if counters >= 100 and counters < 1000:
                incentive = meta.distribute_incentive_2()
            if counters >= 1000 and counters < 2100:
                incentive = []
                spends = []
                avg = []

                for i, a in enumerate(worlds_all_agents):
                    average_action = sum([x.item() for x in action[i]]) / len(action[i])
                    print(f" Average :  {average_action} !! ")
                    spending = 1 - (abs(meta.target[i] - average_action))
                    incentive.append(spending)
                    avg.append(meta.target[i] - average_action)
                    spends.append(spending)
                if flag_for_real_action:
                    reward_without_incentive_baseline.append(sum(reward_without_incentive))
                    baseline.append(avg)
                    baseline_spending.append(sum(spends))


            # if counters >= 1000 and counters < 2000:
            #     incentive = meta.distribute_incentive_4()


                # average_action = sum([x.item() for x in action[0]]) / len(action[0])
                # print(f" Average :  {average_action} !! ")
                # incentive = [1-(abs(0.15-average_action))]

            (obs_, global_state_), (reward,reward_without_incentive), done, _ = world.step(action, flag_for_real_action, incentive, True)
            print(f" I had the following incentive {incentive} ")

            if counters>=2100 and flag_for_real_action:
                #ToDo Single agent tracker
                # incentive_tracker.append(incentive)
                # difference_in_reward.append(reward_without_incentive)
                # averages = sum([x.item() for x in action[0]]) / len(action[0])
                # average.append(averages)
                incentive_tracker.append(sum(incentive))
                difference_in_reward.append(sum(reward_without_incentive))
                averages = []
                for a in range(len(maddpg.actors)):
                    averages.append((meta.target[a] - sum([x.item() for x in action[a]]) / len(action[a])))
                average.append(averages)
            # if not flag_for_incentive and flag_for_real_action  :
            #     before_incentive_reward.append(sum(reward)/4)
            #     difference_in_reward.append(sum(reward_without_incentive)/4)
            # elif flag_for_incentive :
            #     incentive_reward.append(sum(reward)/4)
            #     difference_in_reward.append(sum(reward_without_incentive)/4)
            # if counters>=9000:
            #
            #     exit(1)
            # observation, reward, done, info = world.step(action, None, action_incentive)
            # model.forward(action_incentive, obs_)
            # print("reward: {}\nnew state: {}".format(reward, np.round(obs_, 2)))
            # done = False
            # model.learn(10000, progress_bar=True)
            # action = model.best_action()
            # observation, reward, done, info = env.step(action)
            # model.forward(action, observation)
            # reward = th.FloatTensor(reward).type(FloatTensor)
            # obs_ = np.stack(obs_)
            # obs_ = th.from_numpy(obs_).float()
            # if t != max_steps - 1:
            #     next_obs = obs_
            # else:
            #     next_obs = None
            next_obs = obs_
            next_global_state = global_state_
            # total_reward += reward.sum()
            # rr += reward.cpu()
            # obs_ = np.concatenate([np.expand_dims(obs[2], 0), obs_], 0)
            #ToDo switch around obs -> old version obs,obs_ , new obs_, obs
            maddpg.memory.push(obs, action, obs_, reward, global_state, global_state_)
            maddpg_fake.memory.push(obs, action, obs_, reward, global_state, global_state_) #ToDo fake maddpg
            obs = next_obs
            global_state = next_global_state
            cum_reward.append(sum(reward))
            c_loss, a_loss = maddpg.update_policy(worlds_all_agents, epsilon)
            c_loss, a_loss = maddpg_fake.update_policy(worlds_all_agents, epsilon)#ToDo fake maddpg
            maddpg.episode_done += 1
            maddpg_fake.episode_done += 1

            handle_exploration()
            print(f"actual epsilon {epsilon}")
        maddpg.episode_done += 1
        # print('Episode: %d, reward = %f' % (i_episode, total_reward))
        # reward_record.append(total_reward)

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('training now begins...')
            print('MADDPG on WaterWorld\n' +
                  'scale_reward=%f\n' % scale_reward +
                  'agent=%d' % n_agents +
                  ', coop=%d' % n_coop +
                  ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
                  'food=%f, poison=%f, encounter=%f' % (
                      food_reward,
                      poison_reward,
                      encounter_reward))

    world.close()

