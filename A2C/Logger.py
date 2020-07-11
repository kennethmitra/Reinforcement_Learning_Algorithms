from torch.utils.tensorboard import SummaryWriter
import time

class Logger:
    def __init__(self, run_name=None, refresh_secs=30):
        log_loc = f"runs/{run_name}" if run_name is not None else None
        self.writer = SummaryWriter(log_dir=log_loc, flush_secs=refresh_secs)
        self.last_episode_time = time.perf_counter()

    def log_hparams(self, ENVIRONMENT, SEED, model, LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_COEFF, activation_func,
                    tsteps_per_epoch, normalize_rewards, normalize_advantages, clip_grad, notes, display=True):
        self.writer.add_text("Hyperparams/Environment", ENVIRONMENT, 0)
        self.writer.add_text("Hyperparams/Seed", str(SEED), 0)
        self.writer.add_text("Hyperparams/Model", str(model), 0)
        self.writer.add_text("Hyperparams/Optimizer", str(model.optimizer), 0)
        self.writer.add_text("Hyperparams/Learning_Rate", str(LEARNING_RATE), 0)
        self.writer.add_text("Hyperparams/Discount_Factor", str(DISCOUNT_FACTOR), 0)
        self.writer.add_text("Hyperparams/Entropy_coefficient", str(ENTROPY_COEFF), 0)
        self.writer.add_text("Hyperparams/Activation_Function", str(activation_func), 0)
        self.writer.add_text("Hyperparams/Timesteps_per_epoch", str(tsteps_per_epoch), 0)
        self.writer.add_text("Hyperparams/Normalize_rewards", str(normalize_rewards), 0)
        self.writer.add_text("Hyperparams/Normalize_advantages", str(normalize_advantages), 0)
        self.writer.add_text("Hyperparams/clip_grad", str(clip_grad), 0)
        self.writer.add_text("Hyperparams/Notes", notes, 0)

        if display:
            print('------------------------------Hyperparameters--------------------------------------------------')
            print(f'ENVIRONMENT: {ENVIRONMENT}')
            print(f'SEED: {SEED}')
            print(f'MODEL: {model}')
            print(f'OPTIMIZER: {model.optimizer}')
            print(f'LEARNING_RATE: {LEARNING_RATE}')
            print(f'DISCOUNT_FACTOR: {DISCOUNT_FACTOR}')
            print(f'ENTROPY_COEFF: {ENTROPY_COEFF}')
            print(f'ACTIVATION_FUNC: {activation_func}')
            print(f'NORMALIZE_REWARDS: {normalize_rewards}')
            print(f'NORMALIZE_ADVANTAGES: {normalize_advantages}')
            print(f'CLIP_GRAD: {clip_grad}')
            print(f'NOTES: {notes}')
            print('-----------------------------------------------------------------------------------------------')

    def log_epoch(self, epoch_no, epoch_info):
        self.writer.add_scalar("Loss/Actor_Loss", epoch_info['actor_loss'], epoch_no)
        self.writer.add_scalar("Loss/Critic_Loss", epoch_info['critic_loss'], epoch_no)
        self.writer.add_scalar("Loss/Entropy_Loss", epoch_info['entropy_loss'], epoch_no)
        self.writer.add_scalar("Loss/Entropy", epoch_info['entropy_avg'], epoch_no)
        self.writer.add_scalar("Loss/Total_Loss", epoch_info['total_loss'], epoch_no)
        self.writer.add_scalar("Metrics/Episode_Length", epoch_info['avg_ep_len'], epoch_no)
        self.writer.add_scalar("Metrics/Actual_Timesteps_per_Epoch", epoch_info['epoch_timesteps'], epoch_no)
        self.writer.add_scalar("Metrics/Episodes_per_Epoch", epoch_info['num_episodes'], epoch_no)
        elapsed_time = time.perf_counter() - self.last_episode_time
        self.writer.add_scalar('Time/Time_per_Epoch', elapsed_time, epoch_no)
        self.writer.add_scalar('Metrics/Avg_Raw_Reward', epoch_info['avg_ep_raw_rew'], epoch_no)
        self.last_episode_time = time.perf_counter()

        print(f"Epoch: {epoch_no}, Entropy: {epoch_info['entropy_avg']}, Reward: {epoch_info['avg_ep_raw_rew']}, Time: {elapsed_time}")
