import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=500, batch_size=5000, render=False, discount_factor=0.9):

    epoch_ret_hist = []

    # make environment, check spaces, get obs / act dims0
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_discounted_rtg(rewards, gamma):
        returns_arr = np.zeros_like(rewards, dtype=np.float32)
        rewards_len = len(rewards)
        for i in reversed(range(rewards_len)):
            returns_arr[i] = rewards[i] + (returns_arr[i + 1] * gamma if i + 1 < rewards_len else 0)
        return returns_arr

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            #Custom reward
            #rew *= 100 # Incentivise lasting longer in the game

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += list(get_discounted_rtg(ep_rews, gamma=discount_factor))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens


    # training loop
    for i in range(epochs):
        tic = time.perf_counter()
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        toc = time.perf_counter()
        avg_ret = np.mean(batch_rets)
        avg_len = np.mean(batch_lens)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t time: %.3f'%
                (i, batch_loss, avg_ret, avg_len, toc-tic))
        if (i + 1) % 100 == 0:
            torch.save(logits_net.state_dict(), "{}VPD_RTG-r{}-e{}.save".format(env_name, avg_ret, i))
        epoch_ret_hist.append(float(avg_ret))

    env.close()
    plt.plot(list(range(0, epochs)), epoch_ret_hist)
    plt.xlabel('Epochs')
    plt.ylabel('Return')
    plt.savefig("saves/VPD_RTG_Return1000_2.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    print("Program Arguments: ", args)
    train(env_name=args.env_name, render=args.render, lr=args.lr, epochs=args.epoch, batch_size=5000, hidden_sizes=[64], discount_factor=0.98)
