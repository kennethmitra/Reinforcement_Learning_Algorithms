import torch

class Buffer:
    def __init__(self):

        # Stores one entry per time step
        self.tstep = []
        self.obs = []
        self.act = []
        self.logp = []
        self.val = []
        self.rew = []
        self.entropy = []
        self.disc_rtg_rews = []

        # One entry per episode
        self.per_episode_rews = []
        self.per_episode_length = []

    def record(self, timestep, obs, act, logp, val, rew, entropy):
        self.tstep.append(timestep)
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)
        self.entropy.append(entropy)

    def store_episode_stats(self, episode_rewards, episode_disc_rtg_rews, episode_length):
        self.per_episode_rews.append(torch.tensor(episode_rewards, requires_grad=False).sum().item())
        self.disc_rtg_rews.extend(episode_disc_rtg_rews)
        self.per_episode_length.append(episode_length)

    def get(self):
        data = dict(tstep=self.tstep, obs=self.obs, act=self.act, logp=self.logp, val=self.val, rew=self.rew,
                    entropy=self.entropy, disc_rtg_rews=self.disc_rtg_rews, per_episode_rews=self.per_episode_rews,
                    per_episode_length=self.per_episode_length)
        return data

    def clear(self):
        self.tstep.clear()
        self.obs.clear()
        self.act.clear()
        self.logp.clear()
        self.val.clear()
        self.rew.clear()
        self.entropy.clear()
        self.disc_rtg_rews.clear()
        self.per_episode_rews.clear()
        self.per_episode_length.clear()


# Buffer Unit Tests
if __name__ == '__main__':
    buf = Buffer()
    buf.record(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    buf.record(1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
    data = buf.get()
    print(data)
    print(data['obs'])
    buf.clear()
    data = buf.get()
    print(data)