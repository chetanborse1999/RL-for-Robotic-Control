# RL-for-Robotic-Control
Implementing Reinforcement Learning on 3 DOF arm. Training agent to reach to a specified point in space in Gazebo Simulator.

Environment:
Gazebo Simulator has been used to simulate the 3dof arm. The links of the model has been designed in Autodesk Fusion360 in .stl format and imported in GazeboSim. The joints are described in the UI itself and a plugin has been uploaded to control the joints. 

Plugin:
Plugin has been written in C++. The plugin has been programmed to control three different joints. Each joint imparts a degree of freedom to the arm. The joints are controlled with Proportional control. Velocity control is not achieved yet. The plugin has a provision to recieve angular values via ROS.

ROS(Robot Operating System):
ROS is used to interface the learning script with Gazebo. Custom message types are created to send the angles from learning script to the Simulator. Also the simulator states are read and the position of end effector is tracked by the use of 'topics' which are channels of message transmission.

