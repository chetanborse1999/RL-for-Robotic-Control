# RL-for-Robotic-Control
Implementing Reinforcement Learning on 3 DOF arm. Training agent to reach to a specified point in space in Gazebo Simulator.

### Environment:
Gazebo Simulator has been used to simulate the 3dof arm. The links of the model has been designed in Autodesk Fusion360 in .stl format and imported in GazeboSim. The joints are described in the UI itself and a plugin has been uploaded to control the joints. The environment
![Screenshot from 2019-11-20 19-23-38](https://user-images.githubusercontent.com/36150054/69314802-0d204f80-0c5b-11ea-90af-206022801bcf.png)
##### 3DOF Arm in GazeboSim.

### Plugin:
Plugin has been written in C++. The plugin has been programmed to control three different joints. Each joint imparts a degree of freedom to the arm. The joints are controlled with Proportional control. Velocity control is not achieved yet. The plugin has a provision to recieve angular values via ROS.

### ROS(Robot Operating System):
ROS is used to interface the learning script with Gazebo. Custom message types are created to send the angles from learning script to the Simulator. Also the simulator states are read and the position of end effector is tracked by the use of 'topics' which are channels of message transmission.
![Screenshot from 2019-06-13 00-43-09](https://user-images.githubusercontent.com/36150054/69314826-19a4a800-0c5b-11ea-8ab7-92970436c598.png)
##### Graphical representation of communication in ROS


### Training:
We have implemented DDPG algorithm. The agent performs certain actions based on initial states, moves to another certain state and recieves a reward based on the same. This info is stored in a buffer and used for training in batches. With time, the agent is able to approach the predetermined target.
![mu_losses_3d](https://user-images.githubusercontent.com/36150054/69314844-23c6a680-0c5b-11ea-8f5e-4ac0f87be5e0.png)
##### Mu Losses reflect how the agent gets better at choosing the action.
![q_losses_3d](https://user-images.githubusercontent.com/36150054/69314846-245f3d00-0c5b-11ea-9501-d99fd6a13548.png)
##### Q losses shows the improvement in choosing magnitude of the action.
