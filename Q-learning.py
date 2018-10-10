import gym
import numpy as np
import random
import tensorflow as tf
"""
Q-learning about FrozenLake-v0 
author:likai
time:2018-10-10
"""
#Step 1:Create the environment
env = gym.make("FrozenLake-v0")

#Step 2: Creat the Q-table and initialize it
action_size = env.action_space.n       #  4
state_size = env.observation_space.n   #  16

print("action-size=",action_size)
print("state-size=",state_size)
qtable=np.zeros((state_size,action_size))
print(qtable)


#Step 3: create the hyperparameters

total_episodes = 15000      #Total eposides   
learning_rate = 0.8			#Learning rate
max_steps = 99              #Max steps per episode
gamma = 0.95                #Discpunting rate

# Exploration parameters
epsilon = 1.0               #Exploration rate
max_epsilon = 1.0           #Exploration probability at start
min_epsilon = 0.01			#Minium exploration probability
decay_rate = 0.005          #Exponential decay rate for exploration prob


#Step 4: The Q learning algorithm

rewards = []
#2 for life or until learning is stopped 
for episode in range(total_episodes):
	state = env.reset()
	# State  为整数值，为状态的索引.0-15
	print("state=",state)
	step = 0
	done =False
	total_rewards = 0
	for step in range(max_steps):
		#3 choose an acton a in the current world state(s)
		##Fisrt we randomiz a number
		exp_exp_tradeoff = random.uniform(0, 1)

		## if this number > greater than epsilon --> exploitation(taking the biggest Q value for this state)
		if exp_exp_tradeoff > epsilon:
			action = np.argmax(qtable[state,:])
		#Else doing a random choice --> exploration
		else:
			#print("action_space",env.action_space)
			action = env.action_space.sample()

		print("action=",action) # 0-3
		new_state,reward,done,info = env.step(action)
		print("new_state=",new_state) # 0-15

		# Update Q(s,a):= Q(s,a)+lr[R(s,a)+gamma*max(s',a')-Q(s,a)]
		# qtable[new_state,:]
		qtable[state,action] = qtable[state,action] + learning_rate * (reward + gamma * np.max(qtable[new_state,:]) - qtable[state,action])
		total_rewards += reward

		state = new_state
		if done:
			break
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
	rewards.append(total_rewards)
print("Score over time:"+str(sum(rewards)/total_episodes))
print(qtable)




env.reset()
for episode in range(5):
	state = env.reset()
	step = 0
	done = False
	print("*"*30)
	print("EPISODE",episode)
	for step in range(max_steps):
		action = np.argmax(qtable[state,:])

		new_state,reward,done,info = env.step(action)
		if done:
			env.render()
			print("Number of steps",step)
			break
		state=new_state
env.close()







print("ok!")