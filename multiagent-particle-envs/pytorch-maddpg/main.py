import random
from time import sleep

from MADDPG import MADDPG
import numpy as np
import torch as th

from game.gym_pollinator_game import gymDriver
from params import scale_reward


def make_random_action(all_agents):
    action = []
    for agent in all_agents:
        agent_actions = [th.tensor(random.choice([0,0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100,100,100,100])/100) for _ in range(len(agent.land_cells_owned))]

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
capacity = 1000000
batch_size = 20
n_episode = 20000
max_steps = 100000
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
maddpg = MADDPG(world.n_agents, 12, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):

    obs = world.reset()
    worlds_all_agents = world.agent_processor.all_agents
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

        if epsilon > randy_random:
            action = maddpg.select_action(obs, worlds_all_agents)
            print(f" NEURAL ACTION  {action}")
            action = make_random_action(worlds_all_agents)
            # if buffer_counter % 2 ==0:
            #     buffer_counter=0
            #
            # if buffer_counter==0:
            #     buffer_action = action
            # buffer_counter+=1
            #
            # action = buffer_action

        else:
            # maddpg.batch_size= 32
            action = maddpg.select_action(obs, worlds_all_agents)
            print(f" NEURAL ACTION  {action}")

        # randy_random_2 = random.randint(0,1)
        # if randy_random_2 ==0:
        #     action = [[th.tensor(0.2)]]
        # else:
        #     action = [[th.tensor(0.6)]]
        # choice = random.uniform(0, 1)#random.choice([0.1, 0.2,0.3,0.4,0.5,0.6])
        # action = [[th.tensor(choice).float()]]
        print(f" action for the game{action}")
        # action = th.from_numpy(action_np)
        # obs_, reward, done, _ = world.step(action, randy_random_2)
        obs_, reward, done, _ = world.step(action)

        # reward = th.FloatTensor(reward).type(FloatTensor)
        # obs_ = np.stack(obs_)
        # obs_ = th.from_numpy(obs_).float()
        # if t != max_steps - 1:
        #     next_obs = obs_
        # else:
        #     next_obs = None
        next_obs = obs_
        # total_reward += reward.sum()
        # rr += reward.cpu()
        # obs_ = np.concatenate([np.expand_dims(obs[2], 0), obs_], 0)

        maddpg.memory.push(obs, action, obs_, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy(worlds_all_agents, epsilon)
        maddpg.episode_done += 1
        if maddpg.episode_done >= batch_size:
            if epsilon < 0.1:
                epsilon=0.1
            else:
                epsilon-=1e-3
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
