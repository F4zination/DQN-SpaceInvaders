#!pip install ale-py
#!pip uninstall torch torchvision -y
#!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#!pip install gymnasium[atari]
#!pip install matplotlib

import argparse

import torch
print("CUDA" if torch.cuda.is_available() else "cpu")

# ## SCORING in Space invaders 
# 
# The SPACE INVADERS are worth 5, 10, 15, 20, 25, 30 points in
# the first through sixth rows respectively. (See diagram.) The
# point value of each target stays the same as it drops lower on
# the screen. Each complete set of SPACE INVADERS is worth 630
# points.
# 
# 
# taken from https://atariage.com/manual_html_page.php?SoftwareLabelID=460

import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import ale_py
import random
import math
import numpy as np
from collections import defaultdict

gym.register_envs(ale_py)


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and take max over last 2 frames.
    This handles flickering sprites in Atari games.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        # Buffer to store last 2 observations for max pooling
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Store in buffer (keep last 2 frames)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        # Take element-wise max over last 2 frames
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info

# CLI args
parser = argparse.ArgumentParser(description="Space Invaders DQN long run")
parser.add_argument(
    "--try",
    dest="try_num",
    type=int,
    default=None,
    help="Optional try/restart number to tag output (e.g., video folder).",
)
args = parser.parse_args()

# Training configuration
training_period = 5000           # Record video every 250 episodes
env_name = "ALE/SpaceInvaders-v5" # has a default obs type of rgb, 4 frames are skipped and the repeat action probability is 0.25

# Set up logging for episode statistics
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create the environment
# mode=1 because otherwise the bullets are invisible!!! BUG ON GYM SIDE jissue #524
# frameskip=1 to disable built-in frame skipping (we use MaxAndSkipEnv instead)
# repeat_action_probability=0 for deterministic frame skipping
env = gym.make(env_name, render_mode="rgb_array", obs_type="grayscale", mode=1, frameskip=1, repeat_action_probability=0)

# Apply MaxAndSkipEnv to handle flickering sprites (takes max over last 2 frames)
env = MaxAndSkipEnv(env, skip=4)

video_folder_suffix = f"_try{args.try_num}" if args.try_num is not None else ""
video_folder = f"space_invaders{video_folder_suffix}"

# Record videos periodically (every 250 episodes)
env = RecordVideo(
    env,
    video_folder=video_folder,
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0  # Only record every 250th episode
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)


print(f"Videos will be recorded every {training_period} episodes")
print(f"Videos saved to: {video_folder}/")


from gymnasium.wrappers import FrameStackObservation, ResizeObservation

# ENV Changes
FRAME_STACK_SIZE = 4
RESIZE_X= 84
RESIZE_Y= 84


# This significantly reduces the input size from 210x160
env = ResizeObservation(env, (RESIZE_X, RESIZE_Y))

# This will stack the frame in order to provide the temporal information.
stacked_env = FrameStackObservation(env, stack_size=FRAME_STACK_SIZE)

import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(
        self,
        in_channels,
        action_space
        ):
        super().__init__()

        self.conv_stack = nn.Sequential(
            # Layer 1: 32 filters, 8x8 kernel, stride 4
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Layer 2: 64 filters, 4x4 kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Layer 3: 64 filters, 3x3 kernel, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer fully-connected input size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, RESIZE_X, RESIZE_Y)
            dummy_out = self.conv_stack(dummy)
            self.num_features = dummy_out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc(x)
        return x



# %%
from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# %%
from torch.utils.tensorboard import SummaryWriter


class SpaceInvaderAgent:

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        epsilon_decay_steps: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
        frame_stacking: int = FRAME_STACK_SIZE,
        device: str | None = None,
        log_dir: str = "runs/space_invaders_dqn" # Added log_dir
    ):
        self.env = env

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir)
        self.total_updates = 0 # Track global updates for the loss plot

        # Q-Network to represent the current policy
        self.policy_network = QNetwork(in_channels=frame_stacking, action_space=env.action_space.n).to(self.device)

        self.target_network = QNetwork(in_channels=frame_stacking, action_space=env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor  # How much we care about future rewards also known as gamma

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        # initialize a replay buffer
        self.memory = ReplayMemory(1_000_000)
        self.batch_size = 120

        # Track learning progress
        self.training_error = []

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        """Convert a single stacked observation to a float32 tensor in NCHW."""
        obs = np.array(obs)  # handles FrameStackObservation / LazyFrames
        # Expected shapes:
        # - channels-first: (stack, H, W)
        # - channels-last:  (H, W, stack)
        if obs.ndim != 3:
            raise ValueError(f"Expected 3D obs (stack,H,W) or (H,W,stack), got shape {obs.shape}")

        if obs.shape[0] == FRAME_STACK_SIZE:
            # (stack, H, W)
            pass
        elif obs.shape[-1] == FRAME_STACK_SIZE:
            # (H, W, stack) -> (stack, H, W)
            obs = np.moveaxis(obs, -1, 0)
        else:
            raise ValueError(f"Can't infer channel dimension from obs shape {obs.shape}")

        tensor = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)

        return tensor / 255

    def get_action(self, obs) -> int:
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        obs_t = self._obs_to_tensor(obs).unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            q = self.policy_network(obs_t)
            return int(torch.argmax(q, dim=1).item())

    def update(self):
        if len(self.memory) < 50_000:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

        # Convert each observation robustly to (C,H,W) then stack -> (B,C,H,W)
        state_batch = torch.stack([self._obs_to_tensor(s) for s in state_batch], dim=0)
        next_state_batch = torch.stack([self._obs_to_tensor(s) for s in next_state_batch], dim=0)

        action_batch = torch.as_tensor(action_batch, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.as_tensor(done_batch, dtype=torch.float32, device=self.device)

        # Compute Q-values for current states
        q_values = self.policy_network(state_batch).gather(1, action_batch).squeeze(1)
        state_action_values = self.policy_network(state_batch)
        avg_q_value = state_action_values.max(dim=1)[0].mean().item()

        wandb.log({
            "avg_q_value": state_action_values.mean().item(),
            "max_q_value": state_action_values.max().item(),
            })

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.discount_factor * max_next_q_values * (1.0 - done_batch)

        loss = nn.HuberLoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10)  # Add this
        self.optimizer.step()

        # Log Loss to TensorBoard
        self.writer.add_scalar("Loss/train", loss.item(), self.total_updates)
        self.total_updates += 1

        self.training_error.append(float(loss.item()))

    # Add a method to close the writer
    def close(self):
        self.writer.close()

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, start_epsilon - (global_steps / self.epsilon_decay_steps) * (start_epsilon - final_epsilon))

    # Add this inside the SpaceInvaderAgent class
    def save_checkpoint(self, episode, filename="dqn_space_invaders.pth"):
        """Saves the policy network, optimizer, and current epsilon."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, filename)
        print(f"--> Checkpoint saved at episode {episode}")

    def load_checkpoint(self, filename):
        """Loads a previously saved checkpoint."""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"--> Loaded checkpoint from episode {checkpoint['episode']}")
            return checkpoint['episode']
        return 0



# %%
import wandb

# Training hyperparameters
learning_rate = 0.0002
n_episodes = 100_000        # Number of runs
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
EPSILON_DECAY_STEPS = 300_000

agent = SpaceInvaderAgent(
    env=stacked_env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_decay_steps=EPSILON_DECAY_STEPS,
    final_epsilon=final_epsilon,
)


# Initialize WandB
wandb.init(
    project="space-invaders-dqn",
    config={
        "learning_rate": learning_rate,
        "n_episodes": n_episodes,
        "batch_size": agent.batch_size,
        "epsilon_decay": epsilon_decay,
        "gamma": 0.99,
        "frame_stack": FRAME_STACK_SIZE,
        "device": str(agent.device),
    }
)

wandb.watch(agent.policy_network, log="all", log_freq=100)

steps_in_episode = []
global_steps = 0
highest_reward = 0.0  # Track the highest reward achieved

for episode in range(n_episodes):
    state, info = stacked_env.reset()
    step_counter = 0
    episode_reward = 0.0
    last_lives = 3
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = stacked_env.step(action)
        done = terminated or truncated

        episode_reward += reward

        step_counter += 1
        global_steps += 1 
        clipped_reward = np.clip(reward, -1, 1)
        current_lives = info.get('lives', 3)
        life_lost = current_lives < last_lives 
        agent.memory.push(state, action, next_state, clipped_reward, float(terminated or life_lost))

        last_lives = current_lives

        if global_steps % 4 == 0:
            agent.update()

        state = next_state

        if global_steps % 7_000 == 0:
            agent.target_network.load_state_dict(agent.policy_network.state_dict())
            agent.target_network.eval()
            print(f"Episode {episode} - Step {step_counter}: Updated target network.")


    agent.decay_epsilon()
    steps_in_episode.append(step_counter)

    # Update highest reward if current episode achieved a new high
    if episode_reward > highest_reward:
        highest_reward = episode_reward

    # --- WandB Logging ---
    wandb.log({
        "loss": agent.training_error[-1] if agent.training_error else 0,
        "epsilon": agent.epsilon,
        "episode_reward": episode_reward,
        "highest_reward": highest_reward,
        "episode": episode,
        "global_step": global_steps,
        "steps_per_episode": step_counter,

    })

    # --- Save Model every 500 episodes ---
    if episode % 500 == 0:
        agent.save_checkpoint(episode)



# Close the writer when finished
agent.close()
# Finish Wandb
wandb.finish()



# %%
# python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(agent, env, steps_per_episode):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Episode Rewards
    if hasattr(env, 'return_queue') and len(env.return_queue) > 0:
        rewards = list(env.return_queue)
        ax1.plot(rewards, label='Reward per Episode', alpha=0.3, color='blue')
        if len(rewards) >= 10:
            rolling_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            ax1.plot(rolling_avg, label='Rolling Avg (10 ep)', color='darkblue')
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend()

    # Training Loss
    if len(agent.training_error) > 0:
        ax2.plot(agent.training_error, label='Loss', alpha=0.3, color='orange')
        if len(agent.training_error) >= 50:
            rolling_loss = np.convolve(agent.training_error, np.ones(50)/50, mode='valid')
            ax2.plot(rolling_loss, label='Rolling Avg (50 updates)', color='red')
        ax2.set_title("Training Loss")
        ax2.set_xlabel("Update Step")
        ax2.set_ylabel("Loss (MSE)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No loss data yet.", ha='center', va='center')

    # Steps per Episode
    if steps_per_episode:
        ax3.plot(steps_per_episode, label='Steps per Episode', alpha=0.3, color='green')
        if len(steps_per_episode) >= 10:
            rolling_steps = np.convolve(steps_per_episode, np.ones(10)/10, mode='valid')
            ax3.plot(rolling_steps, label='Rolling Avg (10 ep)', color='darkgreen')
        ax3.set_title("Steps per Episode")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Steps")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No steps data yet.", ha='center', va='center')

    plt.tight_layout()
    plt.show()

# Call with the recorded list:
plot_training_results(agent, env, steps_in_episode)


