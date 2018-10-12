"""
learn from movan
refrence:https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-policy-gradient-softmax1/
"""


import numpy as np 
import tensorflow as tf  

np.random.seed(1) 
tf.set_random_seed(1)


class PolicyGradient:
	def __init__(
		self,
		n_actions,
		n_features,
		learning_rate=0.01,
		reward_decay=0.95,
		output_graph=False,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay


		self.ep_bos,self.ep_as,self.ep_rs = [],[],[]

		self._build_net()

		self.sess = tf.Session()

		if output_graph:
			tf.summary.FileWriter("logs/",self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

	def _build_net(self):
		with tf.name_scope("inputs"):
			self.tf_obs = tf.placeholder(tf.float32,[None,self.n_features],name="observations")
			self.tf_acts = tf.placeholder(tf.float32,[None,],name="actions_num")
			self.tf_vt = tf.placeholder(tf.float32,[None,],name="actions_value")

		layer = tf.layers.dense(
			inputs=self.tf_obs, 
			units=10,
			activation=tf.nn.tanh,
			)

