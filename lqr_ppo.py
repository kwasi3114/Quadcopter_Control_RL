import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from scipy.linalg import eigvalsh

# Define the PPO Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.fc(state)
        std = torch.exp(self.log_std)
        return mean, std

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_epsilon=0.2, c1=0.5, c2=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Coefficient for value loss
        self.c2 = c2  # Coefficient for entropy bonus

    def select_action(self, state):
        #mean, std = self.policy(state)
        mean, std = self.policy.forward(state)
        dist = MultivariateNormal(mean, torch.diag(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        #print("Mean: " + str(mean))
        #print("Std: " + str(std))
        #print("Dist" + str(dist))
        #print("Action: " + str(action))
        #print("Log Prob: " + str(log_prob))
        return action.detach().numpy(), log_prob

    def compute_advantages(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = []
        advantage = 0
        for delta in reversed(deltas):
            advantage = delta + self.gamma * advantage
            advantages.insert(0, advantage)
        return torch.tensor(advantages)

    def update(self, trajectories):
        for trajectory in trajectories:
            states, actions, rewards, log_probs_old, values, next_values, dones = trajectory

            # Compute advantages
            advantages = self.compute_advantages(rewards, values, next_values, dones)
            returns = advantages + values

            # Compute new log probs and values
            means, stds = self.policy(states)
            dist = MultivariateNormal(means, torch.diag(stds))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Ratio for PPO
            ratios = torch.exp(log_probs - log_probs_old)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            # Value loss
            value_loss = ((returns - values) ** 2).mean()

            # Entropy bonus
            entropy_bonus = entropy.mean()

            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy_bonus

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def make_positive_semi_definite(M):
    """
    Ensures the matrix M is positive semi-definite by adjusting eigenvalues.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 0, None)  # Force non-negative eigenvalues
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def make_positive_definite(M):
    """
    Ensures the matrix M is positive definite by slightly perturbing eigenvalues.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-6, None)  # Ensure eigenvalues are > 0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def normalize_matrices(Q, R):
    """
    Normalize Q and R to avoid extreme scales.
    """
    Q_norm = Q / (np.linalg.norm(Q) + 1e-6)
    R_norm = R / (np.linalg.norm(R) + 1e-6)
    return Q_norm, R_norm

# Environment Interaction
def environment_step(Q, R, action):
    """
    Apply an action to modify Q and R, then run the environment and calculate reward.
    """
    Q_new, R_new = modify_matrices(Q, R, action)
    #Q_new = make_positive_semi_definite(Q_new)  # Ensure Q is valid
    R_new = make_positive_definite(R_new)      # Ensure R is valid
    #Q_new, R_new = normalize_matrices(Q_new, R_new)

    #print("Action: " + str(action))
    #print(action)
    #print("Q (old): " + str(Q))
    #print("Q (new): " + str(Q_new))
    #print("R (old): " + str(R))
    #print("R (new): " + str(R_new))

    #try:
    #  position_error = fullRun(Q_new, R_new)  # Execute environment with modified Q, R
    #  reward = -position_error  # Reward is negative position error
    #except Exception as e:
      # Penalize invalid actions or errors
    #  print(f"Error in fullRun: {e}")
    #  reward = -100
    #  Q_new, R_new = Q, R  # Revert to previous matrices

    position_error = fullRun(Q_new, R_new)
    reward = -position_error  # Negative position error as reward
    return reward, Q_new, R_new

def modify_matrices(Q, R, action):
    """
    Modify Q and R matrices based on the action.
    Action is assumed to be a vector where:
    - action[0] modifies Q (leading coefficient)
    - action[1] modifies R (leading coefficient)
    """
    Q_new = Q * abs(10 + action[0])  # Modify Q's leading coefficient
    R_new = R * abs(10 + action[1])  # Modify R's leading coefficient
    return Q_new, R_new

# Main Training Loop
def train_ppo():
    #input_dim = 10  # Example state dimension (you can define this as needed)
    input_dim = 2
    action_dim = 2  # Two actions: modifying Q and R
    agent = PPOAgent(state_dim=input_dim, action_dim=action_dim)

    # Initialize Q and R matrices
    Q = 10 * np.diag([1, 1, 1, 1, 1, 1])
    R = 10 * np.eye(4)

    episodes = 1000
    for episode in range(episodes):
        #state = np.zeros(input_dim)  # Placeholder for the state
        state = [10, 10]
        #trajectories = []

        for t in range(200):  # Steps per episode
            # Select action from the policy
            #print("Iteration Number " + str(t+1) + " of 5")
            action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float32))

            # Environment step
            reward, Q, R = environment_step(Q, R, action)

            # Collect trajectory data (customize based on your PPO implementation)
            # Add code to store states, actions, rewards, log_probs, etc.

            # Example: Update state here if needed
            #state = np.zeros(input_dim)  # Update with your simulation logic
            state = [abs(10+action[0]), abs(10+action[1])]

        # Update PPO agent with collected trajectories
        # Replace `trajectories` with actual trajectory data
        #agent.update(trajectories)
        print("Q: " + str(Q))
        print("R: " + str(R))

        # Print episode progress
        print(f"Episode {episode+1}: Reward {reward}")
        #if episode % 10 == 0:
            #print(f"Episode {episode}: Reward {reward}")



train_ppo()