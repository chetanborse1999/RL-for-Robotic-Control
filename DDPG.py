import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
import math
import gym
from datetime import datetime
from SRA.arm_env import ArmEnv
from ReplayBuffer import ReplayBuffer


def NeuralNet(input_layer, layer_sizes,hidden_activation = tf.nn.relu,output_activation = None):
	for layer in layer_sizes[:-1]:
		input_layer = tf.layers.dense(input_layer,units = layer,activation = hidden_activation)
	output_layer = tf.layers.dense(input_layer,units = layer_sizes[-1],activation = output_activation)
	return output_layer


def getVars(scope):

	return [x for x in tf.global_variables() if scope in x.name]


def createNetworks(s,a, num_actions,action_max, hidden_sizes = (300,), hidden_activation = tf.nn.relu, output_activation = tf.tanh):
	# 1. To multiply the output with , scaled_q
	with tf.variable_scope('mu'):
		mu = action_max*NeuralNet(s, list(hidden_sizes)+[num_actions],hidden_activation,output_activation)
		# scaled_mu = tf.multiply(mu, action_max)
	with tf.variable_scope('q'):
		input_concat = tf.concat([s,a],axis = -1)
		q = tf.squeeze(NeuralNet(input_concat, list(hidden_sizes)+ [1] , hidden_activation, None), axis = 1)		

	with tf.variable_scope('q',reuse = True):
		input_concat = tf.concat([s,mu],axis = -1)
		q_mu = tf.squeeze(NeuralNet(input_concat, list(hidden_sizes)+ [1] , hidden_activation, None), axis = 1)		

	return mu, q , q_mu



def DDPG(ac_kwargs=dict(),
    seed=0,
    save_folder=None,
    num_train_episodes=100,
    test_agent_every=5,
    replay_size=int(1e6),
    gamma=0.99, 
    decay=0.995,
    mu_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=1000, 
    action_noise=0.1,
    max_episode_length=100):
	
	#Set random seeds for reproduction of results
	tf.set_random_seed(seed)
	# np.random.seed(int(seed))

	# create environment instances
	env = ArmEnv((1,1,1))
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]


	# Assuming lower and upper action bounds are equal
	# action_max = env.action_space.high
	action_max = env.action_space.high



	# Create Tensorflow PLACEHOLDERS (neural network inputs), feeds in batches
	################ Note- ORDER OF PLACEHOLDERS MUST MATCH ORDER OF FEED_DICT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  ###############
	S = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # state
	A = tf.placeholder(dtype=tf.float32, shape=(None, num_actions)) # action	
	R = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
	S2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # next state
	D = tf.placeholder(dtype=tf.float32, shape=(None,)) # done



	# Create Main networks 
	with tf.variable_scope('main'):
		mu,q,q_mu = createNetworks(S,A, num_actions,action_max, **ac_kwargs)


	# Target Networks
	# NOTE - FOR TRAINIG Q-NETWORK(CRITIC), WE ONLY NEED THE 'TARGET q_mu'
	# Everything else is not needed

	with tf.variable_scope('target'):
		_,_,q_mu_target = createNetworks(S2,A, num_actions, action_max, **ac_kwargs)



	# Experience Replay memory
	replay_buffer1 = ReplayBuffer(buffer_size = replay_size)
	replay_buffer2 = ReplayBuffer(buffer_size = replay_size)
	replay_buffer3 = ReplayBuffer(buffer_size = replay_size)
	replay_buffers = [replay_buffer1, replay_buffer2, replay_buffer3]

	TARGETS = [(0.283, 1.060, 0.1), (0.283, 1.060, 0.5), (0.283, 1.060, 1.0)]


	# Target value for the Q-network loss
	# We use stop_gradient to tell Tensorflow not to differentiate
	# q_mu_targ wrt any params
	# i.e. consider q_mu_targ values constant
	#  BELLMAN EQUATION. This is important to keep the params of the Differentiator constant while optimizing the Generator NN.
	q_target = tf.stop_gradient(R + gamma * (1 - D) * q_mu_target)


	# DDPG LOSSES (Doubt)

	mu_loss  = -tf.reduce_mean(q_mu)    # The q_mu value itself is the loss for the actor network. It is the optimized action in that state.
	q_loss = tf.reduce_mean((q-q_target)**2)

	# Train each MAIN network separately
	mu_optimizer = tf.train.AdamOptimizer(learning_rate=mu_lr)
	q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
	mu_train_op = mu_optimizer.minimize(mu_loss, var_list=getVars('main/mu'))    # To process the gradient, separately use Compute_gradients and Apply_gradients.
	q_train_op = q_optimizer.minimize(q_loss, var_list=getVars('main/q'))	


	# Use soft updates to UPDATE THE TARGET NETWORKS

	target_update = [tf.assign(v_targ, decay*v_targ + (1 - decay)*v_main) for v_main, v_targ in zip(getVars('main'), getVars('target'))]
	#  ^ THIS DON'T WORK with tf.group
	# target_update = [v_targ.assign(tf.multiply(v_targ,decay) + tf.multiply(v_main,(1-decay)) ) 
	# 				 for v_targ,v_main in zip(tf.trainable_variables(scope= 'target'), tf.trainable_variables(scope = 'main'))]



	# Copy main network params to target networks
	target_init = [tf.assign(v_targ, v_main)
    	for v_main, v_targ in zip(getVars('main'), getVars('target'))
    ]


    # Create Session, Initialise the variables and Copy 'main' to 'target' variables
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(target_init)


	# ACTION FUNCTION
	def get_action (state, noise):
		# print("State is",state)
		action = sess.run(mu,feed_dict = {S: state.reshape(1,-1)})[0]   # The [0] at the end is due to the format of MU - "mu actions - a1: [[ 2.]]"
		# a = sess.run(mu, feed_dict={X: s.reshape(1,-1)})[0]
		
		action += noise* np.random.randn(num_actions)
		return np.clip(action,-action_max,action_max)    # Imma stupid soab for putting 'a' instead of 'action'  <-- dafuq?



	# TEST THE AGENT WITH ENV.RENDER FOR 5 EPISODES.
	test_returns = []
	def test_agent(num_episodes=5):
		t0 = datetime.now()
		n_steps = 0
		for j in range(num_episodes):
			s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
			while not (d or (episode_length == max_episode_length)):
				# Take deterministic actions at test time (noise_scale=0)
				# test_env.render()
				s, r, d, _ = test_env.step(get_action(s, 0))
				episode_return += r
				episode_length += 1
				n_steps += 1
				# print("states",s)
			print('test return:', episode_return, 'episode_length:', episode_length)
			test_returns.append(episode_return)
		print("test steps per sec:", n_steps / (datetime.now() - t0).total_seconds())


	# MAIN LOOP ( WHERE EXPLORATION IS ALSO PRESENT)

	returns = []
	q_losses = []
	mu_losses = []
	num_steps = 0 	
	saver = tf.train.Saver()

	for x in range(len(TARGETS)):
		env, test_env = ArmEnv(TARGETS[x]), ArmEnv(TARGETS[x])
		for i_episode in range(num_train_episodes):

			# Reset env
			s, episode_return, episode_length, d = env.reset(), 0, 0, False

			while not (d or (episode_length == max_episode_length)):
			# For the first `start_steps` steps, use randomly sampled actions
			# in order to encourage exploration.
				if num_steps > start_steps:
					a = get_action(s, action_noise)
				else:
					a = env.action_space.sample()
				
				# Keep track of the number of steps done
				num_steps += 1
				if num_steps == start_steps:
					print("USING AGENT ACTIONS NOW")

				# Step the env
				s2, r, d, _ = env.step(a)
				# print('reward', r)			
				episode_return += r
				episode_length += 1
				print('target_pos', TARGETS[x], 'current_pos', s2)
				# Ignore the "done" signal if it comes from hitting the time
				# horizon (that is, when it's an artificial terminal signal
				# that isn't based on the agent's state)
				d_store = False if episode_length == max_episode_length else d

				# Store experience to replay buffer
				replay_buffers[x].add(s, a, r, s2, d_store)

				# Assign next state to be the current state on the next round
				s = s2
				print('num_steps', num_steps, 'i_episode', i_episode)

			# PERFORM THE UPDATES FOR EQUAL NUMBER OF EPISODES.
			for _ in range(episode_length):
				batch = replay_buffers[x].sample_batch(batch_size)
				# print("Batch DICTIONARY ORDER: ", batch)   # ORDER - S2,S,A,R,D , It may change every time 
				# print("s shape is :",S.shape)	
				# print("state batch :", batch['S'])	
				feed_dict= {
				S : batch['S'],
				S2 : batch['S2'],
				A : batch['A'],
				R : batch['R'],
				D: batch['D']
				}
				# print("Feed dict order",feed_dict)
				# Q Network UPDATE ,(why sess run q)
				ql, _ , _ = sess.run([q_loss,q,q_train_op], feed_dict)
				q_losses.append(ql)

				# POLICY NETWORK (MU) UPDATE,
				# AND TARGET NETWORK UPDATE
				mul, _ , _ = sess.run([mu_loss,mu_train_op,target_update], feed_dict)
				mu_losses.append(mul)

			print ("Episode:", i_episode + 1, "return or reward:", episode_return, "episode_length:" , episode_length)
			returns.append(episode_return)

			# Test agent every '25' episodes for '5' episodes,
			if i_episode == num_train_episodes-1:
				test_agent()

	save_path = saver.save(sess, "/home/chetanborse1999/model.ckpt")



	np.savez('ddpg_results.npz',train = returns, test = test_returns, q_losses = q_losses, mu_losses = mu_losses)
	print("SAVED!!!!!!!!!!")

	plt.plot(q_losses)
	plt.title("q_losses")
	plt.show()

	plt.plot(mu_losses)
	plt.title("mu_losses")
	plt.show()




if __name__ =='__main__':
	import argparse
	start = datetime.now()
	# Yet to add argument parsers
	# TARGET = (0.2846, 0.0696, 3.0463)
	# make_env  = lambda: ArmEnv(TARGET)
	gamma = 0.99
	seed = 0
	save_folder = 'ddpg_monitor'
	num_train_episodes = 200

	DDPG(ac_kwargs= dict(hidden_sizes=[300]*1),gamma= gamma,seed= seed, save_folder= save_folder, num_train_episodes= num_train_episodes)

	print((datetime.now()-start).seconds)


