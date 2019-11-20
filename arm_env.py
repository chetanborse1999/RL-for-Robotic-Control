'''

1) __init__ function
2) step function
3) reset function

'''
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import GetJointProperties
from gazebo_pkg.msg import Angles
import numpy as np
import time
import sys

class Space():
	def __init__(self, rows, columns, low, high):
		self.shape = [rows, columns]
		self.low = low
		self.high = high
	def sample(self):
		sampled = [np.random.uniform(self.low, self.high), np.random.uniform(self.low, self.high), np.random.uniform(self.low, self.high)]
		return np.array(sampled)

class ArmEnv():

	def __init__(self, target):
		print('__init__5')
		self.net_reward = 0
		self.episode_no = 0
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.get_end_effector_pos = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
		self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.get_joint_properties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
		self.target = target
		self.max_angle = 3.14
		self.min_angle = -3.14
		self.n_joints = 3
		self.observation_space = Space(3, 1, None, None)
		self.action_space = Space(self.n_joints, 1, self.min_angle, self.max_angle)


	def step(self, action):
		# print('step')
		self.unpause_sim()
		self.set_action(action)
		# self.pause_sim()
		time.sleep(2)
		obs = self.get_obs()
		done = self.is_done(obs)
		reward = self.calc_reward(obs, done)
		self.net_reward += reward
		info = {'episode_no:':self.episode_no, 'observation:':obs, 'curr_reward':reward, 'net_reward':self.net_reward, 'is_done':done}
		return obs, reward, done, info

	def reset(self):
		# print('reset')
		self.reset_sim()
		self.update_episode() #update episode no and set net reward to zero
		obs = self.get_obs()
		return obs

	def close(self, reason):
		# rospy.signal_shutdown(reason)
		sys.exit()

	def unpause_sim(self):
		# print('unpausing sim')
		self.unpause()

	def pause_sim(self):
		# print('pausing sim')
		self.pause()

	def set_action(self, action):
		# print('setting action:', action)
		# self.pause_sim()
		self.publisher(action)
		self.unpause_sim()

	def get_obs(self):
		end_effector_pos = self.get_end_effector_pos(link_name='test::link_0_0', reference_frame='test::link_0')
		angle1 = self.get_joint_properties(joint_name='link_0_JOINT_0')
		angle2 = self.get_joint_properties(joint_name='link_1_JOINT_1')
		angle3 = self.get_joint_properties(joint_name='link_2_JOINT_3')
		angle1 = angle1.position[0]
		angle2 = angle2.position[0]
		angle3 = angle3.position[0]
		x = end_effector_pos.link_state.pose.position.x
		y = end_effector_pos.link_state.pose.position.y
		z = end_effector_pos.link_state.pose.position.z
		return np.array([x, y, z])

	def is_done(self, observation):
		tolerance = 0.1
		x_min = self.target[0]-tolerance
		x_max = self.target[0]+tolerance
		y_min = self.target[1]-tolerance
		y_max = self.target[1]+tolerance
		z_min = self.target[2]-tolerance
		z_max = self.target[2]+tolerance
		# print(x_min, x_max, y_min, y_max)
		if (observation[0]>x_min and observation[0]<x_max):
			if (observation[1]>y_min and observation[1]<y_max):
				if (observation[2]>z_min and observation[2]<z_max):
					print('DONE DONE DONE!!!!!')
					return True
				else:
					return False
			else:
				return False
		else:
			return False

	def calc_reward(self, observation, done):
		distance = self.dist(observation, self.target)
		reward = -distance
		if done == True:
			reward += 1
		return reward


	def dist(self, observation, target):
		return np.sqrt((target[0]-observation[0])**2 + (target[1]-observation[1])**2 + (target[2]-observation[2])**2)

	def reset_sim(self):
		# print('reset')
		self.pause_sim()
		self.reset_simulation()
		self.publisher((0.0, -3.14, 0))
		self.unpause_sim()
		# time.sleep(1)

	def update_episode(self):
		self.episode_no +=1
		self.net_reward = 0

	def publisher(self, angles):
		pub = rospy.Publisher('/test/vel_cmd', Angles, queue_size=1, latch=True)
		rospy.init_node('set_angle', anonymous=True, disable_signals=True)
		msg = Angles()
		msg.angle1 = angles[0]
		msg.angle2 = self.map_to_joints(angles[1], 0, 3.14)
		msg.angle3 = angles[2]
		pub.publish(msg)
		time.sleep(1)

	def map_to_joints(self, value, min, max):
		return ((value+3.14)/6.28 * (max-min) + min)