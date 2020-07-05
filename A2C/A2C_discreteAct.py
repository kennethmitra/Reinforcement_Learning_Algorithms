import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym
import time
from gradflow_check import *


# Info on A2C http://www.cs.cmu.edu/~rsalakhu/10703/Lectures/Lecture_PG2.pdf
class ActorCritic(torch.nn.Module):
    def __init__(self, obs_cnt, action_cnt, activation_func):
        """
        Construct neural network(s) for actor and critic
        :param obs_cnt: Number of components in an observation
        :param action_cnt: Number of possible actions
        :param activation_func: Shared among all layers except output layers
        """
        super(ActorCritic, self).__init__()
        self.obs_cnt = obs_cnt
        self.action_cnt = action_cnt
        self.activation_func = activation_func
        # self.inputLayer = torch.nn.Linear(obs_cnt, 64)
        # self.actor_layer = torch.nn.Linear(64, 64)
        # self.actor_layer2 = torch.nn.Linear(64, action_cnt)
        # self.critic_layer = torch.nn.Linear(64, 64)
        # self.critic_layer2 = torch.nn.Linear(64, 1)

        # Separate Actor and Critic Networks
        self.actor_layer1 = torch.nn.Linear(obs_cnt, 32)
        self.actor_layer2 = torch.nn.Linear(32, 32)
        self.actor_layer3 = torch.nn.Linear(32, action_cnt)

        self.critic_layer1 = torch.nn.Linear(obs_cnt, 32)
        self.critic_layer2 = torch.nn.Linear(32, 32)
        self.critic_layer3 = torch.nn.Linear(32, 1)

    def forward(self, obs):
        """
        Compute action distribution and value from an observation
        :param obs: observation with len obs_cnt
        :return: Action distrition (Categorical) and value (tensor)
        """
        obs = torch.from_numpy(obs).float()

        # intermed = self.activation_func(self.inputLayer(obs))
        # action_intermed = self.activation_func(self.actor_layer(intermed))
        # actionLogits = self.actor_layer2(action_intermed)
        # critic_intermed = self.activation_func(self.critic_layer(intermed))
        # value = self.critic_layer2(critic_intermed)

        # Separate Actor and Critic Networks
        actor_intermed = self.activation_func(self.actor_layer1(obs))
        actor_intermed = self.activation_func(self.actor_layer2(actor_intermed))
        actionLogits = self.actor_layer3(actor_intermed)

        critic_intermed = self.activation_func(self.critic_layer1(obs))
        critic_intermed = self.activation_func(self.critic_layer2(critic_intermed))
        value = self.critic_layer3(critic_intermed)


        action_dist = Categorical(logits=actionLogits)
        return action_dist, value


def get_discounted_rtg(rewards, gamma):
    returns_arr = torch.zeros(len(rewards))
    rewards_len = len(rewards)
    for i in reversed(range(rewards_len)):
        returns_arr[i] = rewards[i] + (returns_arr[i + 1] * gamma if i + 1 < rewards_len else 0)
    #returns_arr = (returns_arr - returns_arr.mean()) / returns_arr.std() # Normalize rewards
    return list(returns_arr)


def train(env, epochs=10, t_per_epoch=5000, lr=0.01, gamma=0.999, seed=123, renderMode=True):
    torch.manual_seed(seed)
    env.seed(seed)
    print("Running environment", env)
    print("obs_cnt:", env.observation_space.shape[0])
    print("Action cnt:", env.action_space.n)

    model = ActorCritic(obs_cnt=env.observation_space.shape[0], action_cnt=env.action_space.n, activation_func=F.relu)
    print("Model:", model)
    model.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    print("Model.optimizer:", model.optimizer)

    for epoch in range(epochs):
        tic = time.perf_counter()
        #print("Epoch", epoch, "--------------------------------------------------------------------------------------")

        # Epoch specific variables
        epoch_obs = []
        epoch_acts = []
        epoch_pred_values = [] # Predicted V_pi(s) for each observation
        epoch_weights = [] # RTG weights
        epoch_logprobs = [] # Log probability of each action taken (used in loss calculation)
        epoch_rets = []
        epoch_lens = []
        render_current = True

        # Episode specific variables
        obs = env.reset()
        ep_rewards = []
        done = False

        # Collect data on t_per_epoch time steps
        while True:

            # Render first episode of each epoch
            if renderMode and render_current:
                env.render()

            action_dist, value = model.forward(obs)
            action = action_dist.sample()

            # print("Obs:", obs)
            # print("Action_dist:", action_dist)
            # print("Action", action.item())
            # print("Predicted Value:", value)

            # save observation and action
            epoch_obs.append(obs.copy())
            epoch_acts.append(action)
            epoch_pred_values.append(value)
            epoch_logprobs.append(action_dist.log_prob(action))

            # Take action according to policy
            obs, reward, done, _ = env.step(action.item())

            # save reward
            ep_rewards.append(reward)

            if done:
                # Calculate episode return and length
                ep_ret, ep_len = sum(ep_rewards), len(ep_rewards)
                epoch_rets.append(ep_ret)
                epoch_lens.append(ep_len)

                # Only render first episode of epoch
                render_current = False

                # Save discounted Rewards-to-Go
                epoch_weights += get_discounted_rtg(ep_rewards, gamma)

                obs = env.reset()
                ep_rewards = []
                done = False

                # End epoch if we've collected enough experience
                if len(epoch_obs) > t_per_epoch:
                    break

        # Compute advantages
        epoch_advantages = []
        for i in range(len(epoch_acts)):
            epoch_advantages.append(epoch_weights[i].detach() + gamma * (epoch_pred_values[i + 1] if i+1 < len(epoch_acts) else 0) - epoch_pred_values[i])
            #epoch_advantages.append(epoch_weights[i]-epoch_pred_values[i])

        # Compute Loss
        epoch_logprobs = torch.stack(epoch_logprobs)
        epoch_advantages = torch.stack(epoch_advantages)
        epoch_weights = torch.stack(epoch_weights).detach()
        epoch_pred_values = torch.stack(epoch_pred_values)
        epoch_lens = torch.FloatTensor(epoch_lens)

        # First, we find the actor loss
        actor_loss = -(epoch_logprobs * epoch_advantages.detach()).mean()
        # Then the critic MSE loss
        critic_loss = (epoch_weights - epoch_pred_values).pow_(2).mean()
        #critic_loss = epoch_advantages.pow(2).mean()
        # Since we have one optimizer, we can just add the two losses together
        total_loss = actor_loss + critic_loss


        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), "saves/{}A2C-r{}-e{}.save".format('pongRam-', epoch_weights.mean(), epoch))

        # Update Actor and Critic
        model.optimizer.zero_grad()
        total_loss.backward()
        # if (epoch + 1) % 10 == 0:
        #     plot_grad_flow(model.named_parameters())
        # for n, p in model.named_parameters():
        #     if (p.requires_grad) and ("bias" not in n) and ("critic" in n):
        #         a = list(p)
        #         print(a)
        #         break
        model.optimizer.step()


        toc = time.perf_counter()
        print('epoch: %3d \t actor_loss: %.3f \t critic_loss: %.3f \t total_loss: %.3f \t advantage: %.3f \t return: %.3f \t ep_len: %.3f \t time: %.3f' %
              (epoch, actor_loss, critic_loss, total_loss, epoch_advantages.mean(), epoch_weights.mean(), epoch_lens.mean(), toc - tic))

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    train(env, epochs=5000, t_per_epoch=5000, seed=512, renderMode=True)
    env.close()