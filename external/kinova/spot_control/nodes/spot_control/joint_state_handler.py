
import rospy

import kinova_msgs.msg
import actionlib


def joint_angle_client(arm, angle_set):
    """Send a joint angle goal to the action server."""
    action_address = '/' + arm.prefix + 'driver/joints_action/joint_angles'
    client = actionlib.SimpleActionClient(action_address,
                                          kinova_msgs.msg.ArmJointAnglesAction)
    client.wait_for_server()

    goal = kinova_msgs.msg.ArmJointAnglesGoal()

    goal.angles.joint1 = angle_set[0]
    goal.angles.joint2 = angle_set[1]
    goal.angles.joint3 = angle_set[2]
    goal.angles.joint4 = angle_set[3]
    goal.angles.joint5 = angle_set[4]
    goal.angles.joint6 = angle_set[5]
    goal.angles.joint7 = angle_set[6]

    client.send_goal(goal)
    if client.wait_for_result(rospy.Duration(20.0)):
        return client.get_result()
    else:
        print('>> the joint angle action timed-out')
        client.cancel_all_goals()
        return None


def joint_velocity_client(arm, velocity_set):
    """Send a joint angle velocity to the action server."""
    topic_address = '/' + arm.prefix + 'driver/in/joint_velocity'
    pub = rospy.Publisher(topic_address, kinova_msgs.msg.JointVelocity)

    try:

        goal = kinova_msgs.msg.JointVelocity()

        goal.joint1 = velocity_set[0]
        goal.joint2 = velocity_set[1]
        goal.joint3 = velocity_set[2]
        goal.joint4 = velocity_set[3]
        goal.joint5 = velocity_set[4]
        goal.joint6 = velocity_set[5]
        goal.joint7 = velocity_set[6]

        pub.publish(goal)

    except rospy.ROSInterruptException:
        raise ValueError("Error Publishing joint velocities")
