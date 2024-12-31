import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size

        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer= list()
        self.max_memory_buffer = 2000

        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        self.model = Sequential([
            Dense(units=24,input_dim=state_size, activation = 'relu'),
            Dense(units=24,activation = 'relu'),
            Dense(units=action_size, activation = 'linear')
        ])
        self.model.compile(loss="mse",
                      optimizer = Adam(learning_rate=self.lr))

    # The agent computes the action to perform given a state
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action
        #     with the highest Q-value.
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        q_values = self.model.predict(np.reshape(current_state, (1,12)))
        return np.argmax(np.reshape(q_values, (24, )))

    # when an episode is finished, we update the exploration probability using
    # espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        print(self.exploration_proba)

    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done":done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)


    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        #np.random.shuffle(self.memory_buffer)
        #batch_sample = self.memory_buffer[0:self.batch_size]
        batch_sample = np.random.choice(self.memory_buffer, self.batch_size, replace=False)

        # Extract components from the batch
        current_states = np.array([np.reshape(exp["current_state"], (self.state_size,)) for exp in batch_sample])
        next_states = np.array([np.reshape(exp["next_state"], (self.state_size,)) for exp in batch_sample])
        actions = np.array([exp["action"] for exp in batch_sample])
        rewards = np.array([exp["reward"] for exp in batch_sample])
        dones = np.array([exp["done"] for exp in batch_sample])

        # Compute Q-values for current and next states
        q_current_states = self.model.predict(current_states)  # Shape: (batch_size, 24)
        q_next_states = self.model.predict(next_states)        # Shape: (batch_size, 24)

        # Copy Q-values to update targets
        q_targets = q_current_states.copy()

        # Update Q-values for actions taken
        for i in range(self.batch_size):
          if dones[i]:
            q_targets[i, actions[i]] = rewards[i]
          else:
            q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next_states[i])

        self.model.fit(current_states, q_targets, batch_size=self.batch_size, verbose=0)

        # We iterate over the selected experiences
        #for experience in batch_sample:
        #    # We compute the Q-values of S_t
        #    q_current_state = np.reshape(self.model.predict(np.reshape(experience["current_state"], (1,12))), (24, ))
        #    # We compute the Q-target using Bellman optimality equation
        #    q_target = experience["reward"]
        #    q_target = q_target + self.gamma*np.max(np.reshape(self.model.predict(np.reshape(experience["next_state"], (1,12))), (24, )))
        #    q_current_state[0][experience["action"]] = q_target
        #    # train the model
        #    self.model.fit(experience["current_state"], q_current_state, verbose=0)

def increment(value):
  return value + 0.1

def decrement(value):
  return value - 0.1

def take_action(current_state, action_number):
  #print("Action Number: " + str(action_number))
  #print("Current state: " + str(current_state))
  if action_number % 2 == 0:
    #inc = increment(current_state[action_number // 2])
    #print("Increment: " + str(inc))
    #print("Pre-change value: " + str(current_state[action_number // 2]))
    current_state[action_number // 2] = increment(current_state[action_number // 2])
    #print("Post-change value: " + str(current_state[action_number // 2]))
    return current_state
  else:
    #dec = decrement(current_state[action_number // 2])
    #print("Decrement: " + str(dec))
    #print("Pre-change value: " + str(current_state[action_number // 2]))
    current_state[action_number // 2] = decrement(current_state[action_number // 2])
    #print("Post-change value: " + str(current_state[action_number // 2]))
    return current_state

def env_step(action_state):
  Q = action_state[0] * np.diag([action_state[1],
                                  action_state[2],
                                  action_state[3],
                                  action_state[4],
                                  action_state[5],
                                  action_state[6]])

  R = current_state[7] * np.diag([action_state[8],
                                  action_state[9],
                                  action_state[10],
                                  action_state[11]])

  #print("Action state: " + str(action_state))
  #print("Q: \n" + str(Q))
  #print("R: \n" + str(R))

  position_error = fullRun(Q, R)
  reward = -position_error
  return current_state, reward, Q, R

#def is_done(iteration):
#  return iteration >= max_iteration_ep

def reset():
  return start_state

def plot_rewards(reward_history, title="Reward History"):
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

start_state = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# We create our gym environment
# env = gym.make("CartPole-v1")
# We get the shape of a state and the actions space size
state_size = 12
action_size = 24
# Number of episodes to run
n_episodes = 20
# Max iterations per epiode
max_iteration_ep = 30
# We define our agent
agent = DQNAgent(state_size, action_size)
total_steps = 0
current_state = start_state
current_state = np.array(current_state)
reward_history = []
#print("Starting state: " + str(current_state))

# We iterate over episodes
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit
    #  with the input layer of the DNN
    #current_state = init_state
    #current_state = np.array([current_state])
    #try:
      for step in range(max_iteration_ep):
          total_steps = total_steps + 1
          iter_start_state = current_state
          # the agent computes the action to perform
          action_number = agent.compute_action(current_state)
          action_state = take_action(current_state, action_number)
          # the envrionment runs the action and returns
          # the next state, a reward and whether the agent is done
          next_state, reward, Q_mat, R_mat = env_step(action_state)
          next_state = np.array(next_state)

          done = False
          if step+1 == max_iteration_ep:
            done = True

          # We sotre each experience in the memory buffer
          agent.store_episode(current_state, action_number, reward, next_state, done)
          print("Current State: " + str(iter_start_state))
          print("Action: " + str(action_number))
          print("Reward: " + str(reward))
          print("Next State: " + str(next_state))
          print("Q \n" + str(Q_mat))
          print("R \n" + str(R_mat))
          print("-------------------------------------------------------------")

          # if the episode is ended, we leave the loop after
          # updating the exploration probability
          if done:
              agent.update_exploration_probability()
              reward_history.append(reward)
              break
          current_state = next_state

    #except Exception as e:
      #print(f"Error in training: {e}. Ending episode prematurely and resetting state.")
      #current_state = reset()
      #agent.update_exploration_probability()

    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
      if total_steps >= agent.batch_size:
        agent.train()


plot_rewards(reward_history, title="Reward History")