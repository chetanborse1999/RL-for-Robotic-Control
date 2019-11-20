#include "boost/bind.hpp"
#include "gazebo/gazebo.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "stdio.h"
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
// #include "std_msgs/Float32.h"
#include <gazebo_pkg/Angles.h>



// In the real file these quotes are greater-than and less-than but I
// don't know how to get that to show up in my question

namespace gazebo
{
  class ModelControl : public ModelPlugin
  {
  public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;

      // Store the pointers to the joints
      this->joint1 = this->model->GetJoint("link_0_JOINT_0");
      this->joint2 = this->model->GetJoint("link_1_JOINT_1");
      this->joint3 = this->model->GetJoint("link_2_JOINT_3");
      
      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&ModelControl::OnUpdate, this, _1));

      // Initialize ros, if it has not already bee initialized.
if (!ros::isInitialized())
{
  int argc = 0;
  char **argv = NULL;
  ros::init(argc, argv, "gazebo_client",
      ros::init_options::NoSigintHandler);
}

// Create our ROS node. This acts in a similar manner to
// the Gazebo node
this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

// Create a named topic, and subscribe to it.
ros::SubscribeOptions so =
  ros::SubscribeOptions::create<gazebo_pkg::Angles>(
      "/" + this->model->GetName() + "/vel_cmd",
      1,
      boost::bind(&ModelControl::OnRosMsg, this, _1),
      ros::VoidPtr(), &this->rosQueue);
this->rosSub = this->rosNode->subscribe(so);

// Spin up the queue helper thread.
this->rosQueueThread =
  std::thread(std::bind(&ModelControl::QueueThread, this));
    }

    // Called by the world update start event
  public: void OnUpdate(const common::UpdateInfo & /*_info*/)
    {
      // Apply a small linear velocity to the model.
      //this->model->SetLinearVel(math::Vector3(.03, 0, 0));
      
      // Apply angular velocity to joint
      // this->joint1->SetVelocity(0, 1.5707);
      // this->joint1->SetPosition(0, 1.5);
      
      // this->joint1->SetParam("max_force", 0, 9999999999);
      double currAngle1 = this->joint1->GetAngle(0).Radian();
      double currAngle2 = this->joint2->GetAngle(0).Radian();
      double currAngle3 = this->joint3->GetAngle(0).Radian();
      printf("Current angle1: %f\n", currAngle1);
      printf("Current angle2: %f\n", currAngle2);
      printf("Current angle3: %f\n", currAngle3);
      // setAngle1 = 4.71+1.57/2;
      // setAngle2 = 1.57;

      this->joint1->SetVelocity(0, 5*(setAngle1-currAngle1));
      
      if (currAngle1<setAngle1+0.05 && currAngle1>setAngle1-0.05)
      {
        // this->joint1->SetPosition(0, setAngle1);
        this->joint1->SetVelocity(0, 0);
      }

      this->joint2->SetVelocity(0, 5*(setAngle2-currAngle2));
      
      if (currAngle2<setAngle2+0.05 && currAngle2>setAngle2-0.05)
      {
        // this->joint2->SetPosition(0, setAngle2);
        this->joint2->SetVelocity(0, 0);
      }

      this->joint3->SetVelocity(0, 5*(setAngle3-currAngle3));
      
      if (currAngle3<setAngle3+0.05 && currAngle3>setAngle3-0.05)
      {
        // this->joint1->SetPosition(0, setAngle1);
        this->joint3->SetVelocity(0, 0);
      }
       
      if ((currAngle1<setAngle1+0.075 && currAngle1>setAngle1-0.075)&&(currAngle2<setAngle2+0.075 && currAngle2>setAngle2-0.075)&&(currAngle3<setAngle3+0.075 && currAngle3>setAngle3-0.075))
      {
        this->joint1->SetPosition(0, setAngle1);
        this->joint2->SetPosition(0, setAngle2);
      }

      // if (currAngle1<rad+0.1 && currAngle1>rad-0.1)
      // {
      //   this->joint1->SetVelocity(0, -1.5707);
      //   this->joint1->SetPosition(0, rad);
      // }
    }

    /// \brief Handle an incoming message from ROS
/// \param[in] _msg A float value that is used to set the velocity
/// of the Velodyne.
public: void OnRosMsg(const gazebo_pkg::AnglesConstPtr &msg1)
{

  // this->joint1->SetVelocity(0, _msg->data);
  setAngle1 = msg1->angle1;
  setAngle2 = msg1->angle2;
  setAngle3 = msg1->angle3;

  
  // setAngle2 = _msg2->data; 

  // else
  // {
  //   this->joint1->SetVelocity(0, 0);
  //   this->joint1->SetPosition(0, currAngle1);
  // }
  // if (currAngle1<angle+0.1 && currAngle1>angle-0.1)
  //     {
  //       this->joint1->SetPosition(0, angle);
  //       this->joint1->SetVelocity(0, 0);
  //     } 
}

/// \brief ROS helper function that processes messages
private: void QueueThread()
{
  static const double timeout = 0.01;
  while (this->rosNode->ok())
  {
    this->rosQueue.callAvailable(ros::WallDuration(timeout));
  }
}

    // Maybe I want to keep track of time?
    common::Time last_update_time_;

    // Pointer to the model
  private: physics::ModelPtr model;

  // double setAngle;
  double setAngle1;
  double setAngle2;
  double setAngle3;

    // Pointer to the update event connection
  private: event::ConnectionPtr updateConnection;

    // Pointers to joints
  physics::JointPtr joint1;
  physics::JointPtr joint2;
  physics::JointPtr joint3;

  /// \brief A node use for ROS transport
private: std::unique_ptr<ros::NodeHandle> rosNode;

/// \brief A ROS subscriber
private: ros::Subscriber rosSub;

/// \brief A ROS callbackqueue that helps process messages
private: ros::CallbackQueue rosQueue;

/// \brief A thread the keeps running the rosQueue
private: std::thread rosQueueThread;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelControl)
}
