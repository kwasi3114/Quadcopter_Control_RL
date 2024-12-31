# Quadcopter_Control_RL
Repository for the development of a reinforcement learning program to tune the LQR controller for a quadcopter. Goal of the RL algorithm is to minimize position error. 

Initially, Proximal Policy Optimization (PPO) was going to be used, but Deep Q-Learning was instead chosen for the final implementation due to it suiting the nature of the problem more. Both implementations are in this repository, though the PPO implementation has some bugs and does not run correctly. The DQN implementation is mostly complete save for some minor bugs. 
