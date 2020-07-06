import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import count
import gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time


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

        # Separate Actor and Critic Networks
        self.actor_layer1 = torch.nn.Linear(obs_cnt, 32)
        self.actor_layer3 = torch.nn.Linear(32, action_cnt)

        self.critic_layer1 = torch.nn.Linear(obs_cnt, 32)
        self.critic_layer3 = torch.nn.Linear(32, 1)

        self.episodeMem = []
        self.episodeRewards = []
        self.episodeSumRewards = 0

        self.writer = SummaryWriter()

    def forward(self, obs):
        """
        Compute action distribution and value from an observation
        :param obs: observation with len obs_cnt
        :return: Action distrition (Categorical) and value (tensor)
        """
        obs = torch.from_numpy(obs).float()

        # Separate Actor and Critic Networks
        actor_intermed = self.activation_func(self.actor_layer1(obs))
        actionLogits = self.actor_layer3(actor_intermed)

        critic_intermed = self.activation_func(self.critic_layer1(obs))
        value = self.critic_layer3(critic_intermed)

        action_dist = Categorical(logits=actionLogits)
        return action_dist, value

    def sample_action(self, obs):
        """
        Given an observation, predict action distribution and value and sample action
        :param obs: observation from env.step() or env.reset()
        :return: sampled action, log prob of sampled action, value calculated by critic
        """
        action_dist, value = model.forward(obs)
        action = action_dist.sample()

        return action, action_dist.log_prob(action), value

    def record(self, timestep, obs, action, logprob, value):
        self.episodeMem.append((timestep, obs, action, logprob, value))

    def record_reward(self, reward, episode):
        self.episodeRewards.append(reward)
        self.episodeSumRewards += reward

    def clear_episode_mem(self):
        self.episodeMem.clear()
        self.episodeRewards.clear()
        self.episodeSumRewards = 0

    def update_tensorboard(self, episode):
        model.writer.add_scalar('Metrics/Raw_Reward', self.episodeSumRewards, episode)

    def learn_from_experience(self, gamma, episode, normalize_returns=True):
        # Calculate discounted Rewards-To-Go (returns)
        returns = []
        running_sum = 0
        for r in self.episodeRewards[::-1]:
            running_sum = r + gamma*running_sum
            returns.insert(0, running_sum)

        # Don't need to backprop through returns so I can convert to tensor after calculation
        returns = torch.tensor(returns)

        if normalize_returns:
            returns = (returns - returns.mean()) / returns.std()


        # Sanity Check
        assert len(self.episodeRewards) == len(self.episodeMem)
        assert len(returns) == len(self.episodeMem)

        # Calculate actor and critic loss
        actor_loss = []
        critic_loss = []
        for (timestep, obs, action, logprob, value), return_ in zip(self.episodeMem, returns):
            return_ = torch.tensor([return_])
            advantage = return_ - value
            actor_loss.append(-(logprob * advantage))
            # Why L1 loss? From pytorch doc:
            # It is less sensitive to outliers than the MSELoss and in some cases prevents exploding gradients
            critic_loss.append(F.smooth_l1_loss(return_, value))
            #critic_loss.append(advantage.pow(2))

        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = torch.stack(critic_loss).sum()
        total_loss = actor_loss + critic_loss

        #Update Tensorboard
        self.writer.add_scalar("Loss/Actor_Loss", actor_loss, episode)
        self.writer.add_scalar("Loss/Critic_Loss", critic_loss, episode)
        self.writer.add_scalar("Loss/Total_Loss", total_loss, episode)

        # Perform backprop step
        model.optimizer.zero_grad()
        total_loss.backward()
        model.optimizer.step()

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rewards')
    ax.plot(steps)

    plt.pause(0.0000001)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.99
    NUM_EPISODES = 100000
    model = ActorCritic(obs_cnt=env.observation_space.shape[0], action_cnt=env.action_space.n, activation_func=F.relu)
    model.optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        episode_rewards = 0

        render = False
        if episode % 100 == 0:
            render = True

        for timestep in count():
            action, logprob, value = model.sample_action(obs)
            model.record(timestep, obs, action, logprob, value)
            obs, reward, done, _ = env.step(action.item())
            model.record_reward(reward, episode=episode)
            episode_rewards += reward

            if render: env.render()
            if done: break
        model.update_tensorboard(episode)
        model.learn_from_experience(gamma=DISCOUNT_FACTOR, normalize_returns=False, episode=episode)
        model.clear_episode_mem()
