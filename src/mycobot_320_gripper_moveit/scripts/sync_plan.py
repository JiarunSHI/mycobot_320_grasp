#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import queue
import threading
import time

import pymycobot
import rospy
from packaging import version
from sensor_msgs.msg import JointState


MAX_REQUIRE_VERSION = "3.5.3"

if version.parse(pymycobot.__version__) > version.parse(MAX_REQUIRE_VERSION):
    raise RuntimeError(
        "The pymycobot version must be <= {}. Current version: {}".format(
            MAX_REQUIRE_VERSION,
            pymycobot.__version__,
        )
    )

from pymycobot.mycobot import MyCobot


ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
GRIPPER_JOINT = "gripper_controller"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def gripper_value_to_radian(value):
    return clamp(float(value), 0.0, 100.0) / 117.0 - 0.7


def gripper_radian_to_value(value):
    return int(round(clamp((float(value) + 0.7) * 117.0, 0.0, 100.0)))


def valid_arm_positions(values):
    if not isinstance(values, (list, tuple)) or len(values) < len(ARM_JOINTS):
        return False
    try:
        return all(
            math.isfinite(float(value))
            for value in values[: len(ARM_JOINTS)]
        )
    except (TypeError, ValueError):
        return False


class MyCobotSerialBridge(object):
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyACM0")
        self.baud = rospy.get_param("~baud", 115200)
        self.arm_speed = rospy.get_param("~speed", 80)
        self.gripper_speed = rospy.get_param("~gripper_speed", 100)
        self.command_topic = rospy.get_param("~joint_state_topic", "planned_joint_states")
        self.feedback_topic = rospy.get_param("~feedback_joint_state_topic", "/joint_states")
        self.feedback_rate = max(0.5, float(rospy.get_param("~joint_state_rate", 5.0)))
        self.gripper_feedback_rate = max(
            0.2,
            float(rospy.get_param("~gripper_state_rate", 2.0)),
        )
        self.gripper_control_enabled = bool(
            rospy.get_param("~gripper_control_enabled", False)
        )
        self.gripper_feedback_enabled = bool(
            rospy.get_param("~gripper_feedback_enabled", False)
        )
        self.command_queue_size = max(
            1,
            int(rospy.get_param("~command_queue_size", 100)),
        )
        initial_gripper_value = rospy.get_param("~initial_gripper_value", 50.0)

        self.command_queue = queue.Queue(maxsize=self.command_queue_size)
        self.last_arm_positions = None
        self.last_gripper_position = gripper_value_to_radian(initial_gripper_value)
        self.have_real_gripper_feedback = False
        self.shutdown_event = threading.Event()

        rospy.loginfo(
            "Connecting to MyCobot on port=%s baud=%s (pymycobot=%s)",
            self.port,
            self.baud,
            pymycobot.__version__,
        )
        self.mc = MyCobot(self.port, self.baud)
        if self.gripper_control_enabled:
            self.mc.set_gripper_mode(0)
            time.sleep(0.5)

        self.feedback_pub = rospy.Publisher(
            self.feedback_topic,
            JointState,
            queue_size=20,
        )
        self.command_sub = rospy.Subscriber(
            self.command_topic,
            JointState,
            self.command_callback,
            queue_size=50,
        )

        self.worker = threading.Thread(target=self.serial_worker)
        self.worker.daemon = True
        self.worker.start()

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("sync_plan command input: %s", self.command_topic)
        rospy.loginfo("sync_plan real feedback output: %s", self.feedback_topic)
        if not self.gripper_control_enabled:
            rospy.loginfo("Gripper control disabled; gripper commands will be ignored.")
        if not self.gripper_feedback_enabled:
            rospy.loginfo(
                "Gripper feedback disabled; /joint_states uses the last commanded gripper position."
            )

    def shutdown(self):
        self.shutdown_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

    def read_joint_positions(self, data):
        positions = [float(value) for value in data.position]
        joints = dict(zip(data.name, positions)) if data.name else {}
        arm_positions = None
        gripper_position = None

        if joints and all(joint in joints for joint in ARM_JOINTS):
            arm_positions = [joints[joint] for joint in ARM_JOINTS]
        elif not joints and len(positions) >= len(ARM_JOINTS):
            arm_positions = positions[: len(ARM_JOINTS)]

        if GRIPPER_JOINT in joints:
            gripper_position = joints[GRIPPER_JOINT]
        elif not joints and len(positions) > len(ARM_JOINTS):
            gripper_position = positions[len(ARM_JOINTS)]
        elif not joints and len(positions) == 1:
            gripper_position = positions[0]

        return arm_positions, gripper_position

    def command_callback(self, data):
        try:
            arm_positions, gripper_position = self.read_joint_positions(data)
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(5.0, "Invalid command JointState: %s", exc)
            return
        if arm_positions is None and gripper_position is None:
            rospy.logwarn_throttle(
                5.0,
                "sync_plan could not find arm joints or %s in names=%s positions=%d",
                GRIPPER_JOINT,
                list(data.name),
                len(data.position),
            )
            return

        command = (arm_positions, gripper_position)
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            try:
                self.command_queue.get_nowait()
                self.command_queue.task_done()
            except queue.Empty:
                pass
            self.command_queue.put_nowait(command)
            rospy.logwarn_throttle(
                2.0,
                "sync_plan command queue is full; dropped the oldest stale path point.",
            )

    def send_command(self, command):
        arm_positions, gripper_position = command

        if arm_positions is not None:
            rospy.logdebug("Sending arm radians: %s", arm_positions)
            self.mc.send_radians(arm_positions, self.arm_speed)

        if self.gripper_control_enabled and gripper_position is not None:
            gripper_value = gripper_radian_to_value(gripper_position)
            rospy.logdebug(
                "Sending gripper radian=%.3f value=%d",
                gripper_position,
                gripper_value,
            )
            try:
                self.mc.set_gripper_value(gripper_value, self.gripper_speed, 1)
            except TypeError:
                self.mc.set_gripper_value(gripper_value, self.gripper_speed)
            self.last_gripper_position = float(gripper_position)

    def read_arm_feedback(self):
        try:
            values = self.mc.get_radians()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to read arm feedback: %s", exc)
            return False

        if not valid_arm_positions(values):
            rospy.logwarn_throttle(5.0, "Invalid arm feedback: %s", values)
            return False

        self.last_arm_positions = [
            float(value) for value in values[: len(ARM_JOINTS)]
        ]
        return True

    def read_gripper_feedback(self):
        try:
            value = self.mc.get_gripper_value()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to read gripper feedback: %s", exc)
            return False

        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            rospy.logwarn_throttle(5.0, "Empty gripper feedback; using last known value.")
            return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, "Invalid gripper feedback: %s", value)
            return False

        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            rospy.logwarn_throttle(5.0, "Out-of-range gripper feedback: %s", value)
            return False

        self.last_gripper_position = gripper_value_to_radian(value)
        self.have_real_gripper_feedback = True
        return True

    def publish_feedback(self):
        if self.last_arm_positions is None:
            return

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ARM_JOINTS + [GRIPPER_JOINT]
        msg.position = self.last_arm_positions + [self.last_gripper_position]
        self.feedback_pub.publish(msg)

    def poll_and_publish_feedback(self, now, next_feedback, next_gripper_feedback):
        if now < next_feedback:
            return next_feedback, next_gripper_feedback

        self.read_arm_feedback()
        next_feedback = now + (1.0 / self.feedback_rate)

        if (
            self.gripper_control_enabled
            and self.gripper_feedback_enabled
            and now >= next_gripper_feedback
        ):
            self.read_gripper_feedback()
            next_gripper_feedback = now + (1.0 / self.gripper_feedback_rate)

        self.publish_feedback()
        return next_feedback, next_gripper_feedback

    def serial_worker(self):
        next_feedback = time.monotonic()
        next_gripper_feedback = time.monotonic()

        while not rospy.is_shutdown() and not self.shutdown_event.is_set():
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                command = None

            if command is not None:
                try:
                    self.send_command(command)
                except Exception as exc:
                    rospy.logerr_throttle(2.0, "Failed to send robot command: %s", exc)
                finally:
                    self.command_queue.task_done()
                next_feedback, next_gripper_feedback = self.poll_and_publish_feedback(
                    time.monotonic(),
                    next_feedback,
                    next_gripper_feedback,
                )
                continue

            now = time.monotonic()
            if now >= next_feedback:
                next_feedback, next_gripper_feedback = self.poll_and_publish_feedback(
                    now,
                    next_feedback,
                    next_gripper_feedback,
                )
                continue

            time.sleep(min(0.01, max(0.001, next_feedback - now)))


def main():
    rospy.init_node("sync_plan")
    MyCobotSerialBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
