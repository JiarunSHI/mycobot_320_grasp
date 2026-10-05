#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import queue
import sys
import threading

import moveit_commander
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def get_plan_trajectory(plan_result):
    if isinstance(plan_result, tuple):
        success = plan_result[0]
        plan = plan_result[1]
    else:
        plan = plan_result
        success = bool(plan.joint_trajectory.points)

    if not success or not plan.joint_trajectory.points:
        return None
    return plan.joint_trajectory


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def gripper_value_to_radian(value):
    return clamp(float(value), 0.0, 100.0) / 117.0 - 0.7


def is_gripper_value(value):
    return 0.0 <= float(value) <= 100.0


def all_nan(values):
    return all(math.isnan(float(value)) for value in values)


class PosePlanJointStatePublisher(object):
    def __init__(self):
        self.command_mode = rospy.get_param("~command_mode", "topic")
        self.command_topic = rospy.get_param("~command_topic", "/mycobot320/target_pose_rpy")
        self.status_topic = rospy.get_param("~status_topic", "/mycobot320/motion_status")
        self.group_name = rospy.get_param("~group", "arm_group")
        self.gripper_group_name = rospy.get_param("~gripper_group", "gripper_group")
        self.gripper_joint = rospy.get_param("~gripper_joint", "gripper_controller")
        self.gripper_control_enabled = bool(
            rospy.get_param("~gripper_control_enabled", False)
        )
        self.reference_frame = rospy.get_param("~reference_frame", "base")
        self.end_effector_link = rospy.get_param("~end_effector_link", "")
        self.joint_state_topic = rospy.get_param("~joint_state_topic", "planned_joint_states")
        self.actual_joint_state_topic = rospy.get_param(
            "~actual_joint_state_topic",
            "/joint_states",
        )
        self.execute = rospy.get_param("~execute", False)
        self.time_scale = rospy.get_param("~time_scale", 1.0)
        self.hold_seconds = rospy.get_param("~hold_seconds", 0.5)
        self.start_delay = rospy.get_param("~start_delay", 2.0)
        self.move_group_wait = rospy.get_param("~move_group_wait", 30.0)
        self.wait_for_subscriber = rospy.get_param("~wait_for_subscriber", True)
        self.subscriber_timeout = rospy.get_param("~subscriber_timeout", 30.0)
        self.command_queue_size = max(
            1,
            int(rospy.get_param("~command_queue_size", 20)),
        )
        self.verify_reached = rospy.get_param("~verify_reached", True)
        self.actual_state_timeout = rospy.get_param("~actual_state_timeout", 15.0)
        self.reached_timeout = rospy.get_param("~reached_timeout", 20.0)
        self.arm_goal_tolerance = rospy.get_param("~arm_goal_tolerance", 0.220)
        self.gripper_goal_tolerance = rospy.get_param("~gripper_goal_tolerance", 0.200)
        self.reached_stable_samples = max(
            1,
            int(rospy.get_param("~reached_stable_samples", 3)),
        )

        self.x = rospy.get_param("~x", 0.20)
        self.y = rospy.get_param("~y", 0.00)
        self.z = rospy.get_param("~z", 0.20)
        self.roll = rospy.get_param("~roll", 0.0)
        self.pitch = rospy.get_param("~pitch", 0.0)
        self.yaw = rospy.get_param("~yaw", 0.0)

        self.max_velocity_scaling = rospy.get_param("~max_velocity_scaling", 0.2)
        self.max_acceleration_scaling = rospy.get_param("~max_acceleration_scaling", 0.2)
        self.planning_time = rospy.get_param("~planning_time", 5.0)

        self.gripper_position = rospy.get_param("~gripper_position", None)
        self.default_gripper_value = self.get_requested_gripper_value()
        self.gripper_value = self.default_gripper_value
        self.command_queue = queue.Queue(maxsize=self.command_queue_size)
        self.actual_joint_lock = threading.Lock()
        self.actual_joint_positions = {}
        self.last_actual_joint_state_time = rospy.Time(0)

        self.pub = rospy.Publisher(self.joint_state_topic, JointState, queue_size=20)
        self.status_pub = rospy.Publisher(
            self.status_topic,
            String,
            queue_size=10,
            latch=True,
        )
        self.actual_joint_sub = rospy.Subscriber(
            self.actual_joint_state_topic,
            JointState,
            self.actual_joint_state_callback,
            queue_size=20,
        )
        rospy.loginfo(
            "Waiting up to %.1f seconds for move_group action server...",
            self.move_group_wait,
        )
        self.group = moveit_commander.MoveGroupCommander(
            self.group_name,
            wait_for_servers=self.move_group_wait,
        )
        self.group.set_pose_reference_frame(self.reference_frame)
        self.group.set_max_velocity_scaling_factor(self.max_velocity_scaling)
        self.group.set_max_acceleration_scaling_factor(self.max_acceleration_scaling)
        self.group.set_planning_time(self.planning_time)

        if self.end_effector_link:
            self.group.set_end_effector_link(self.end_effector_link)

        rospy.loginfo("MoveIt group: %s", self.group_name)
        rospy.loginfo("Planning frame: %s", self.group.get_planning_frame())
        rospy.loginfo("End effector link: %s", self.group.get_end_effector_link())
        rospy.loginfo("JointState output topic: %s", self.joint_state_topic)
        rospy.loginfo("Actual JointState input topic: %s", self.actual_joint_state_topic)
        rospy.loginfo("Motion status topic: %s", self.status_topic)
        rospy.loginfo("execute: %s", self.execute)
        rospy.loginfo("command_mode: %s", self.command_mode)
        rospy.loginfo("gripper_control_enabled: %s", self.gripper_control_enabled)

        self.gripper_group = None
        self.publish_status("IDLE")

    def publish_status(self, status):
        self.status_pub.publish(String(data=status))
        rospy.loginfo("Motion status: %s", status)

    def actual_joint_state_callback(self, msg):
        with self.actual_joint_lock:
            for name, position in zip(msg.name, msg.position):
                self.actual_joint_positions[name] = float(position)
            self.last_actual_joint_state_time = rospy.Time.now()

    def wait_for_actual_arm_state(self):
        deadline = rospy.Time.now() + rospy.Duration(self.actual_state_timeout)
        required_joints = set(self.group.get_active_joints())
        available_joints = set()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.actual_joint_lock:
                available_joints = set(self.actual_joint_positions)
                state_time = self.last_actual_joint_state_time

            if (
                required_joints.issubset(available_joints)
                and state_time != rospy.Time(0)
                and (rospy.Time.now() - state_time).to_sec() < 2.0
            ):
                return True
            rate.sleep()

        missing = sorted(required_joints - available_joints)
        rospy.logerr(
            "No fresh real arm state received on %s; missing joints=%s",
            self.actual_joint_state_topic,
            missing,
        )
        return False

    def should_plan_gripper(self):
        return self.gripper_control_enabled and is_gripper_value(self.gripper_value)

    def get_requested_gripper_value(self):
        gripper_value = float(rospy.get_param("~gripper_value", -1.0))
        gripper_angle = float(rospy.get_param("~gripper_angle", -1.0))

        if is_gripper_value(gripper_value):
            return gripper_value
        if is_gripper_value(gripper_angle):
            return gripper_angle
        return -1.0

    def ensure_gripper_group(self):
        if self.gripper_group is not None:
            return

        self.gripper_group = moveit_commander.MoveGroupCommander(
            self.gripper_group_name,
            wait_for_servers=self.move_group_wait,
        )
        self.gripper_group.set_max_velocity_scaling_factor(self.max_velocity_scaling)
        self.gripper_group.set_max_acceleration_scaling_factor(self.max_acceleration_scaling)
        self.gripper_group.set_planning_time(self.planning_time)
        rospy.loginfo("Gripper group initialized: %s", self.gripper_group_name)

    def build_target_pose(self):
        qx = rospy.get_param("~qx", None)
        qy = rospy.get_param("~qy", None)
        qz = rospy.get_param("~qz", None)
        qw = rospy.get_param("~qw", None)
        if None in (qx, qy, qz, qw):
            qx, qy, qz, qw = quaternion_from_euler(self.roll, self.pitch, self.yaw)

        pose = PoseStamped()
        pose.header.frame_id = self.reference_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = self.z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def plan_arm(self):
        if self.start_delay > 0:
            rospy.loginfo("Waiting %.1f seconds before planning...", self.start_delay)
            rospy.sleep(self.start_delay)
        self.group.set_start_state_to_current_state()
        rospy.loginfo("Current joints: %s", [round(v, 3) for v in self.group.get_current_joint_values()])
        target_pose = self.build_target_pose()

        if self.end_effector_link:
            self.group.set_pose_target(target_pose, self.end_effector_link)
        else:
            self.group.set_pose_target(target_pose)

        rospy.loginfo(
            "Planning %s to x=%.3f y=%.3f z=%.3f in frame %s",
            self.group_name,
            self.x,
            self.y,
            self.z,
            self.reference_frame,
        )

        trajectory = get_plan_trajectory(self.group.plan())
        self.group.clear_pose_targets()

        if trajectory is None:
            rospy.logerr("MoveIt did not find a valid plan.")
            rospy.logerr("Try a different x/y/z or orientation, or set end_effector_link explicitly.")
            return None

        rospy.loginfo("MoveIt plan contains %d trajectory points.", len(trajectory.points))
        return trajectory

    def plan_gripper(self):
        if not self.should_plan_gripper():
            return None

        self.ensure_gripper_group()
        gripper_radian = gripper_value_to_radian(self.gripper_value)
        self.gripper_group.set_start_state_to_current_state()
        self.gripper_group.set_joint_value_target({self.gripper_joint: gripper_radian})

        rospy.loginfo(
            "Planning %s to %s=%.3f rad from gripper_value=%.1f",
            self.gripper_group_name,
            self.gripper_joint,
            gripper_radian,
            self.gripper_value,
        )

        trajectory = get_plan_trajectory(self.gripper_group.plan())
        if trajectory is None:
            rospy.logerr("MoveIt did not find a valid gripper plan.")
            return None

        rospy.loginfo("MoveIt gripper plan contains %d trajectory points.", len(trajectory.points))
        return trajectory

    def plan(self, arm_target_enabled=True):
        trajectories = []

        if arm_target_enabled:
            arm_trajectory = self.plan_arm()
            if arm_trajectory is None:
                return None
            trajectories.append((self.group_name, arm_trajectory))

        if self.should_plan_gripper():
            gripper_trajectory = self.plan_gripper()
            if gripper_trajectory is None:
                return None
            trajectories.append((self.gripper_group_name, gripper_trajectory))

        if not trajectories:
            rospy.logerr("Command has no arm target and no enabled gripper target.")
            return None

        return trajectories

    def parse_command_message(self, msg):
        values = list(msg.data)
        if len(values) < 6:
            rospy.logerr(
                "Target command needs at least 6 values: [x, y, z, roll, pitch, yaw]."
            )
            return None

        if len(values) > 8:
            rospy.logwarn("Target command has extra values; ignoring values after index 7.")

        try:
            pose_values = [float(value) for value in values[:6]]
        except (TypeError, ValueError):
            rospy.logerr("Target command contains a non-numeric xyz/rpy value.")
            return None

        arm_target_enabled = not all_nan(pose_values)
        if arm_target_enabled and not all(math.isfinite(value) for value in pose_values):
            rospy.logerr(
                "Pose values must be finite, or all six pose values must be nan for a gripper-only command."
            )
            return None
        x, y, z, roll, pitch, yaw = pose_values

        gripper_value = self.default_gripper_value
        if len(values) >= 7:
            try:
                candidate = float(values[6])
            except (TypeError, ValueError):
                rospy.logerr("Target command contains a non-numeric gripper value.")
                return None

            if candidate < 0:
                gripper_value = -1.0
            elif is_gripper_value(candidate):
                gripper_value = candidate
            else:
                rospy.logerr("Gripper value must be in 0-100, or negative to disable it.")
                return None

        arm_goal_tolerance = self.arm_goal_tolerance
        if len(values) >= 8:
            try:
                candidate = float(values[7])
            except (TypeError, ValueError):
                rospy.logerr("Target command contains a non-numeric arm goal tolerance.")
                return None
            if not math.isfinite(candidate) or candidate <= 0.0:
                rospy.logerr("Arm goal tolerance must be a positive finite value in radians.")
                return None
            arm_goal_tolerance = candidate

        if not arm_target_enabled and not is_gripper_value(gripper_value):
            rospy.logerr("Gripper-only command requires gripper_value in 0-100.")
            return None

        return (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            gripper_value,
            arm_goal_tolerance,
            arm_target_enabled,
        )

    def apply_target(self, x, y, z, roll, pitch, yaw, gripper_value, arm_target_enabled=True):
        if arm_target_enabled:
            self.x = x
            self.y = y
            self.z = z
            self.roll = roll
            self.pitch = pitch
            self.yaw = yaw
        self.gripper_value = gripper_value

    def execute_target(
        self,
        x,
        y,
        z,
        roll,
        pitch,
        yaw,
        gripper_value,
        arm_goal_tolerance,
        arm_target_enabled=True,
    ):
        self.publish_status("ACCEPTED")
        self.apply_target(
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            gripper_value,
            arm_target_enabled=arm_target_enabled,
        )
        if arm_target_enabled:
            rospy.loginfo(
                "Processing target: x=%.3f y=%.3f z=%.3f r=%.3f p=%.3f y=%.3f gripper=%.1f arm_tolerance=%.3f rad",
                self.x,
                self.y,
                self.z,
                self.roll,
                self.pitch,
                self.yaw,
                self.gripper_value,
                arm_goal_tolerance,
            )
        else:
            rospy.loginfo(
                "Processing gripper-only target: gripper=%.1f",
                self.gripper_value,
            )

        if arm_target_enabled:
            self.publish_status("WAITING_FOR_STATE")
            if not self.wait_for_actual_arm_state():
                self.publish_status("STATE_TIMEOUT")
                return False

        self.publish_status("PLANNING")
        trajectories = self.plan(arm_target_enabled=arm_target_enabled)
        if trajectories is None:
            self.publish_status("PLAN_FAILED")
            return False

        if not self.execute:
            self.publish_status("PLANNED")
            return True

        self.publish_status("EXECUTING")
        expected_positions = self.publish_trajectories(trajectories)
        if expected_positions is None:
            self.publish_status("EXECUTION_FAILED")
            return False

        if self.verify_reached and not self.wait_until_reached(
            expected_positions,
            arm_goal_tolerance,
        ):
            self.publish_status("REACHED_TIMEOUT")
            return False

        self.publish_status("REACHED")
        return True

    def command_callback(self, msg):
        target = self.parse_command_message(msg)
        if target is None:
            return

        try:
            self.command_queue.put_nowait(target)
            rospy.loginfo(
                "Queued target command (%d waiting).",
                self.command_queue.qsize(),
            )
            self.publish_status("QUEUED")
        except queue.Full:
            rospy.logerr(
                "Target queue is full (%d); rejecting the new target.",
                self.command_queue_size,
            )
            self.publish_status("QUEUE_FULL")

    def command_worker(self):
        while not rospy.is_shutdown():
            try:
                target = self.command_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.execute_target(*target)
            except Exception as exc:
                rospy.logerr("Unhandled target execution error: %s", exc)
                self.publish_status("EXECUTION_FAILED")
            finally:
                self.command_queue.task_done()

            if self.command_queue.empty():
                rospy.loginfo("Target queue is empty.")

    def run(self):
        if self.command_mode == "once":
            self.execute_target(
                self.x,
                self.y,
                self.z,
                self.roll,
                self.pitch,
                self.yaw,
                self.gripper_value,
                self.arm_goal_tolerance,
                True,
            )
            return

        if self.command_mode != "topic":
            rospy.logerr("Unsupported command_mode '%s'. Use 'topic' or 'once'.", self.command_mode)
            return

        rospy.Subscriber(
            self.command_topic,
            Float64MultiArray,
            self.command_callback,
            queue_size=self.command_queue_size,
        )
        worker = threading.Thread(target=self.command_worker)
        worker.daemon = True
        worker.start()
        rospy.loginfo(
            "Waiting for target commands on %s as std_msgs/Float64MultiArray.",
            self.command_topic,
        )
        rospy.loginfo(
            "Command format: data=[x, y, z, roll, pitch, yaw, gripper_value, optional_arm_tolerance]."
        )
        rospy.loginfo(
            "Use data=[nan, nan, nan, nan, nan, nan, gripper_value] for a gripper-only command."
        )
        rospy.spin()

    def wait_for_joint_state_subscriber(self):
        if not self.wait_for_subscriber:
            return True

        deadline = rospy.Time.now() + rospy.Duration(self.subscriber_timeout)
        rospy.loginfo(
            "Waiting for a subscriber on %s before publishing trajectory...",
            self.joint_state_topic,
        )
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.pub.get_num_connections() == 0:
            if rospy.Time.now() > deadline:
                rospy.logerr(
                    "No subscriber connected to %s after %.1f seconds.",
                    self.joint_state_topic,
                    self.subscriber_timeout,
                )
                rospy.logerr(
                    "If you want the real robot to move, start sync_plan with sync_joint_state_topic:=%s.",
                    self.joint_state_topic,
                )
                return False
            rate.sleep()

        rospy.sleep(0.2)
        rospy.loginfo(
            "Subscriber connected to %s: %d",
            self.joint_state_topic,
            self.pub.get_num_connections(),
        )
        return True

    def publish_trajectories(self, trajectories):
        if not self.wait_for_joint_state_subscriber():
            return None

        expected_positions = {}
        for group_name, trajectory in trajectories:
            self.publish_trajectory(group_name, trajectory)
            if trajectory.points:
                final_point = trajectory.points[-1]
                expected_positions.update(
                    zip(trajectory.joint_names, final_point.positions)
                )

        return expected_positions

    def angular_error(self, actual, target):
        return abs(math.atan2(math.sin(actual - target), math.cos(actual - target)))

    def wait_until_reached(self, expected_positions, arm_goal_tolerance):
        if not expected_positions:
            return True

        deadline = rospy.Time.now() + rospy.Duration(self.reached_timeout)
        stable_samples = 0
        rate = rospy.Rate(10)
        rospy.loginfo(
            "Waiting up to %.1f seconds for real joint feedback to reach the goal.",
            self.reached_timeout,
        )

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.actual_joint_lock:
                actual_positions = dict(self.actual_joint_positions)

            missing = [
                joint for joint in expected_positions if joint not in actual_positions
            ]
            errors = {}
            if not missing:
                for joint, target in expected_positions.items():
                    errors[joint] = self.angular_error(
                        actual_positions[joint],
                        float(target),
                    )

            within_tolerance = not missing
            for joint, error in errors.items():
                tolerance = (
                    self.gripper_goal_tolerance
                    if joint == self.gripper_joint
                    else arm_goal_tolerance
                )
                if error > tolerance:
                    within_tolerance = False
                    break

            if within_tolerance:
                stable_samples += 1
                if stable_samples >= self.reached_stable_samples:
                    rospy.loginfo("Real joint feedback reached the planned goal.")
                    return True
            else:
                stable_samples = 0
                if missing:
                    rospy.logwarn_throttle(
                        3.0,
                        "Waiting for real joint feedback: missing joints=%s",
                        missing,
                    )
                elif errors:
                    rospy.loginfo_throttle(
                        2.0,
                        "Waiting for goal, maximum joint error=%.3f rad",
                        max(errors.values()),
                    )

            rate.sleep()

        rospy.logerr("Timed out waiting for the real robot to reach the planned goal.")
        return False

    def publish_trajectory(self, group_name, trajectory):
        rospy.loginfo("Publishing trajectory as JointState messages on %s", self.joint_state_topic)
        rospy.loginfo("Publishing %s trajectory with joints: %s", group_name, trajectory.joint_names)
        previous_time = rospy.Duration(0.0)

        for point in trajectory.points:
            if rospy.is_shutdown():
                return

            wait_time = (point.time_from_start - previous_time).to_sec() * self.time_scale
            if wait_time > 0:
                rospy.sleep(wait_time)
            previous_time = point.time_from_start

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = list(trajectory.joint_names)
            msg.position = list(point.positions)

            if (
                self.gripper_control_enabled
                and self.gripper_position is not None
                and not self.should_plan_gripper()
                and self.gripper_joint not in msg.name
            ):
                msg.name.append(self.gripper_joint)
                msg.position.append(float(self.gripper_position))

            self.pub.publish(msg)

        rospy.loginfo("Finished publishing %d trajectory points.", len(trajectory.points))

        if self.hold_seconds > 0 and trajectory.points:
            end_msg = JointState()
            end_msg.header.stamp = rospy.Time.now()
            end_msg.name = list(trajectory.joint_names)
            end_msg.position = list(trajectory.points[-1].positions)
            if (
                self.gripper_control_enabled
                and self.gripper_position is not None
                and not self.should_plan_gripper()
                and self.gripper_joint not in end_msg.name
            ):
                end_msg.name.append(self.gripper_joint)
                end_msg.position.append(float(self.gripper_position))

            stop_time = rospy.Time.now() + rospy.Duration(self.hold_seconds)
            rate = rospy.Rate(10)
            while not rospy.is_shutdown() and rospy.Time.now() < stop_time:
                end_msg.header.stamp = rospy.Time.now()
                self.pub.publish(end_msg)
                rate.sleep()


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("plan_to_pose_joint_state", anonymous=True)

    try:
        node = PosePlanJointStatePublisher()
        node.run()
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
