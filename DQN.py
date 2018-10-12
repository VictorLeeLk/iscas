import gym
from gym.wrappers import Monitor
import itertools
import numpy as np 
import os
import random
import sys
import tensorflow as tf 

if "../" not in sys.path:
	sys.path.append("../")

from lib import plotting
from collections import deque,namedtuple

env = gym.envs.make("Breakout-v0")

VALID_ACTIONS = [0,1,2,3]

class StateProcessor():
	def __init__(self):
		# Build the tensorflow graph
		with tf.variable_scope("state_processor"):
			self.input_state=tf.placeholder(shape=[210,160,3],dtype=tf.unit8)
			self.output=tf.image.rgb_to_grayscale(self.input_state)
			self.output=tf.image.crop_to_bounding_box(self.output,34,0,160,160)
			self.output=tf.image.resize_images(self.output, [84,84],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output=tf.squeeze(self.output)
	def process(self,sess,state):
	"""
	Args:
	    sess:A tensorflow session object
	    state:A [210,160,3] Atari RGB State
	Returns:
	    A processed [84,84] state representing grayscale values
	"""
		return sess.run(self.output,{self.input_state:state})
class Estimator():
	"""
	Q-value Estimator neural network
	this network is used for both the Q-network and the target network
	"""
	def __init__(self,scope="estimator",summaries_dir=None):
		self.scope=scope
		self.summary_writer=None
		with tf.variable_scope(scope):
			self._build_model()
			if summaries_dir;
			summaries_dir = os.path.join(summaries_dir,"summaries_{}".format(scope))
			if not os.path.exists(summaries_dir):
				os.makedirs(summaries_dir)
			self.summary_writer = tf.summary.FileWriter(summaries_dir)
	def _build_model(self):
		"""
		Builds the tensorflow graph
		"""
		# Placeholders for our input
		# Our input are 4 RGB frames of shape 160 160 each
		self.X_pl=tf.placeholder(shape=[None,84,84,4],dtype=tf.unit8,name="X")
		#the TD target value
		self.y_pl=tf.placeholder(shape=[None],dtype=tf.float32,name="y")
		self.actions_pl=tf.placeholder(shape=[None],dtype=tf.int32,name="actions")

		X=tf.to_float(self.X_pl)/255.0
		batch_size=tf.shape(self.X_pl)[0]

		#Three convolutional layers
		conv1 = tf.contrib.layers.conv2d(
			X,32,8,4,activation_fn=tf.nn.relu)
		conv2 = tf.contrib.layers.conv2d(
			conv1,64,4,2,activation_fn=tf.nn.relu)
		conv3 = tf.contrib.layers.conv2d(
			conv2,64,3,1,activation_fn=tf.nn.relu)

		#Fully connected layers
		flattened = tf.contrib.layers.flatten(conv3)
		fc1 = tf.contrib.layers.fully_connected(flattened,512)
		self.predications = tf.contrib.layers.fully_connected(fc1,len(VALID_ACTIONS))

		#Get the predictions for the chosen actions only
		gather_indices = tf.range(batch_size) * tf.shape(self.predications)[1] + self.actions_pl
		self.action_predictions = tf.gather(tf.reshape(self.predications,[-1]),gather_indices)

		#Calculate the loss
		self.losses= tf.squared_difference(self.y_pl,self.action_predictions)
		self.loss = tf.reduce_mean(self.losses)

		#OPtimizer Parameters from original paper
		self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6)
		self.train_op = self.optimizer.minimize(self.loss,global_step=tf.contrib.framework.get_global_step())

		#Summaries for tensorboard

		self.summaries = tf.summary.merge([
			tf.summary.scalar("loss",self.loss),
			tf.summary.histogram("loss_hist",self.losses),
			tf.summary.histogram("q_value_hist", self.predications),
			tf.summary.scalar("max_q_value", tf.reduce_max(self.predications))
		])
	def predict(self,sess,s):
		"""
		predicts action values

		args:
			sess: tensorflow session
			s:state input of shape [batch_size,4,160,160,3]
		returns :
			tensor of shape [batch_size,NUM_VALID_ACTIONS] containing the estimated action values.
		"""
		return sess.run(self.predications,{self.X_pl:s})
	def update(self,sess,s,a,y):
		"""
		update the estimator towards the given targets.
		Args:
			sess:tensorflow session object
			s:state input of shape [batch_size,4,160,160,3]
			a:chosen action of shape [batch_size]
			y:targets of shape [batch_size]
		returns:
			the calculated loss on the batch.
		"""

		feed_dict={self.X_pl:s,self.y_pl:y,self.actions_pl:a}
		summaries,global_step,_,loss=sess.run(
			[self.summaries,tf.contrib.framework.get_global_step(),self.train_op,self.loss],
			feed_dict)
		if self.summary_writer:
			self.summary_writer.add_summary(summaries,global_step)
		return loss
		
		




















