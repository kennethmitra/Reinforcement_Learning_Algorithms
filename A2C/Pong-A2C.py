import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import count
import gym
import matplotlib.pyplot as plt
from Buffer import Buffer
from Logger import Logger


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
        self.actor_layer1 = torch.nn.Linear(obs_cnt, 64)
        self.actor_layer2 = torch.nn.Linear(64, action_cnt)
        torch.nn.init.xavier_uniform_(self.actor_layer1.weight)
        torch.nn.init.xavier_uniform_(self.actor_layer2.weight)

        self.critic_layer1 = torch.nn.Linear(obs_cnt, 64)
        self.critic_layer2 = torch.nn.Linear(64, 1)
        torch.nn.init.xavier_uniform_(self.critic_layer1.weight)
        torch.nn.init.xavier_uniform_(self.critic_layer2.weight)


    def forward(self, obs):
        """
        Compute action distribution and value from an observation
        :param obs: observation with len obs_cnt
        :return: Action distrition (Categorical) and value (tensor)
        """
        obs = torch.from_numpy(obs).float()

        # Separate Actor and Critic Networks
        actor_intermed = self.activation_func(self.actor_layer1(obs))
        actionLogits = self.actor_layer2(actor_intermed)

        critic_intermed = self.activation_func(self.critic_layer1(obs))
        value = self.critic_layer2(critic_intermed)

        action_dist = Categorical(logits=actionLogits)
        return action_dist, value

    def sample_action(self, obs):
        """
        Given an observation, predict action distribution and value and sample action
        :param obs: observation from env.step() or env.reset()
        :return: sampled action, log prob of sampled action, value calculated by critic, entropy of action prob dist
        """
        action_dist, value = model.forward(obs)
        action = action_dist.sample()
        entropy = action_dist.entropy()

        return action, action_dist.log_prob(action), value, entropy

    def save(self, epoch, run_name):
        try:
            torch.save({'epoch': epoch,
                        'optimizer_params': self.optimizer.state_dict(),
                        'model_state': self.state_dict()}, f'./saves/{run_name}-epo{epoch}.save')
        except:
            print('ERROR calling model.save()')

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_params'])
        return checkpoint['epoch']

    def discount_rewards_to_go(self, episode_rewards, gamma):
        # Calculate discounted Rewards-To-Go (returns)
        returns = []
        running_sum = 0
        for r in episode_rewards[::-1]:
            running_sum = r + gamma * running_sum
            returns.insert(0, running_sum)
        return returns

    def learn_from_experience(self, data, entropy_coeff, normalize_returns=True,
                              normalize_advantages=True,
                              clip_grad=True):

        # Sanity Check
        assert len(data['tstep']) == len(data['obs']) == len(data['act']) == len(data['logp']) == len(data['val']) \
               == len(data['rew']) == len(data['entropy']) == len(data['disc_rtg_rews']) == len(data['disc_rtg_rews'])
        assert len(data['per_episode_rews']) == len(data['per_episode_length'])

        # Don't need to backprop through returns
        returns = torch.tensor(data['disc_rtg_rews'])

        if normalize_returns:
            # returns = (returns - returns.mean()) / returns.std()
            returns = (returns) / returns.std()

        # Calculate advantages separately (to apply normalization)
        advantages = []
        for return_, value in zip(returns, data['val']):
            advantages.append(return_ - value)
            #advantages.append(return_)

        advantages = torch.tensor(advantages)

        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std()

        assert len(advantages) == len(data['tstep'])

        # Zero out gradients before calculating loss
        model.optimizer.zero_grad()

        # Calculate actor and critic loss
        actor_loss = []
        critic_loss = []
        for logprob, advantage, return_, value in zip(data['logp'], advantages, returns, data['val']):
            actor_loss.append(-(logprob * advantage))
            # Why L1 loss? From pytorch doc:
            # It is less sensitive to outliers than the MSELoss and in some cases prevents exploding gradients
            critic_loss.append(F.smooth_l1_loss(return_, torch.squeeze(value)))
            # critic_loss.append(advantage.pow(2))

            # Entropy Loss (https://medium.com/@awjuliani/maximum-entropy-policies-in-reinforcement-learning-everyday-life-f5a1cc18d32d)
            # https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

        actor_loss = torch.stack(actor_loss).mean()
        critic_loss = 0.5 * torch.stack(critic_loss).mean()
        entropy_avg = torch.stack(data['entropy']).mean()
        entropy_loss = -(entropy_coeff * entropy_avg)
        total_loss = actor_loss + critic_loss + entropy_loss

        # Perform backprop step
        total_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        model.optimizer.step()

        # Compute info for logging
        avg_ep_len = torch.tensor(data['per_episode_length'], requires_grad=False, dtype=torch.float).mean().item()
        avg_ep_raw_rew = torch.tensor(data['per_episode_rews'], requires_grad=False, dtype=torch.float).mean().item()
        epoch_timesteps = data['tstep'][-1]
        num_episodes = len(data['per_episode_length'])

        # Return logging info
        return dict(actor_loss=actor_loss, critic_loss=critic_loss, entropy_loss=entropy_loss, entropy_avg=entropy_avg,
                    total_loss=total_loss, avg_ep_len=avg_ep_len, avg_ep_raw_rew=avg_ep_raw_rew,
                    epoch_timesteps=epoch_timesteps, num_episodes=num_episodes, advantages=advantages,
                    pred_values=data['val'], disc_rews=returns)


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
    print("-------------------------------GPU INFO--------------------------------------------")
    print('Available devices ', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current cuda device ', device)
    print('Current CUDA device name ', torch.cuda.get_device_name(device))
    print("-----------------------------------------------------------------------------------")
    ENVIRONMENT = 'Pong-v0'
    SEED = 543
    ACTOR_LEARNING_RATE = 0.0007
    CRITIC_LEARNING_RATE = 0.0014
    DISCOUNT_FACTOR = 0.99
    ENTROPY_COEFF = 0.0
    NUM_EPOCHS = 100000
    TIMESTEPS_PER_EPOCH = 50000
    ACTIVATION_FUNC = torch.relu
    NORMALIZE_REWARDS = False
    NORMALIZE_ADVANTAGES = True
    CLIP_GRAD = True
    NUM_PROCESSES = 1
    RUN_NAME = "Pong-A2C"
    NOTES = "Continued run; normalize advantages, clip grad, no entropy coeff"

    torch.manual_seed(SEED)
    env = gym.make(ENVIRONMENT)
    env.seed(SEED)

    model = ActorCritic(obs_cnt=env.observation_space.shape[0], action_cnt=env.action_space.n,
                        activation_func=ACTIVATION_FUNC)
    model.optimizer = torch.optim.Adam(params=[{'params': list(model.actor_layer1.parameters()) + list(model.actor_layer2.parameters()), 'lr': ACTOR_LEARNING_RATE},
                                               {'params': list(model.critic_layer1.parameters()) + list(model.critic_layer2.parameters()), 'lr': CRITIC_LEARNING_RATE}])

    buf = Buffer()
    log = Logger(run_name=None, refresh_secs=30)

    # Load saved weights
    # epoch = model.load("./saves/Pong-A2C-epo1700.save")
    # Override epoch
    start_epoch = 0

    log.log_hparams(ENVIRONMENT=ENVIRONMENT, SEED=SEED, model=model, ACTOR_LEARNING_RATE=ACTOR_LEARNING_RATE,
                    CRITIC_LEARNING_RATE=CRITIC_LEARNING_RATE, DISCOUNT_FACTOR=DISCOUNT_FACTOR,
                    ENTROPY_COEFF=ENTROPY_COEFF, activation_func=ACTIVATION_FUNC,
                    tsteps_per_epoch=TIMESTEPS_PER_EPOCH, normalize_rewards=NORMALIZE_REWARDS,
                    normalize_advantages=NORMALIZE_ADVANTAGES, clip_grad=CLIP_GRAD, notes=NOTES, display=True)

    # Setup env for first episode
    obs = env.reset()
    episode_rewards = []
    episode = 0
    epoch = 0

    # Iterate over epochs
    for epoch in range(start_epoch, NUM_EPOCHS):

        # Render first episode of every Nth epoch
        render = ((epoch % 1) == 0)

        # Continue getting timestep data until reach TIMESTEPS_PER_EPOCH
        for timestep in count():

            # Get action prediction from model
            action, logprob, value, entropy = model.sample_action(obs)

            # Perform action in environment and get new observation and rewards
            new_obs, reward, done, _ = env.step(action.item())

            # Store state-action information for updating model
            buf.record(timestep=timestep, obs=obs, act=action, logp=logprob, val=value, entropy=entropy, rew=reward)

            obs = new_obs
            episode_rewards.append(reward)
            if render: env.render()

            if done:
                render = False

                # Store discounted Rewards-To-Go™
                ep_disc_rtg = model.discount_rewards_to_go(episode_rewards=episode_rewards, gamma=DISCOUNT_FACTOR)
                buf.store_episode_stats(episode_rewards=episode_rewards, episode_disc_rtg_rews=ep_disc_rtg,
                                        episode_length=timestep)

                # Initialize env after end of episode
                obs = env.reset()
                episode_rewards.clear()
                episode += 1

                if timestep >= TIMESTEPS_PER_EPOCH or timestep > 10000:
                    break

        # Save model
        if epoch % 100 == 0:
            try:
                model.save(epoch=epoch, run_name=RUN_NAME)
            except:
                print('ERROR calling model.save()')

        # Train model on epoch data
        epoch_info = model.learn_from_experience(data=buf.get(), normalize_returns=NORMALIZE_REWARDS,
                                                 entropy_coeff=ENTROPY_COEFF, clip_grad=CLIP_GRAD)

        # Log epoch statistics and clear buffer
        log.log_epoch(epoch, epoch_info)
        buf.clear()

    # After Training
    model.save(epoch=epoch, run_name=RUN_NAME)
    env.close()
