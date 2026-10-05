#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import sys
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, String


ACTIVE_MOTION_STATES = {
    "QUEUED",
    "ACCEPTED",
    "WAITING_FOR_STATE",
    "PLANNING",
    "EXECUTING",
}
FAILED_MOTION_STATES = {
    "PLAN_FAILED",
    "EXECUTION_FAILED",
    "REACHED_TIMEOUT",
    "STATE_TIMEOUT",
    "QUEUE_FULL",
}


def normalize_quaternion(q):
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return tuple(value / norm for value in q)


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def quaternion_to_euler(x, y, z, w):
    x, y, z, w = normalize_quaternion((x, y, z, w))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def quaternion_conjugate(q):
    x, y, z, w = normalize_quaternion(q)
    return -x, -y, -z, w


def rotate_vector(q, vector):
    x, y, z, w = normalize_quaternion(q)
    vx, vy, vz = vector

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def quaternion_angle(q1, q2):
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class ArucoPoseController(object):
    def __init__(self):
        self.marker_id = int(rospy.get_param("~marker_id", 23))
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.marker_frame = rospy.get_param(
            "~marker_frame",
            "aruco_marker_%d" % self.marker_id,
        )
        self.command_topic = rospy.get_param(
            "~command_topic",
            "/mycobot320/aruco/target_pose_rpy",
        )
        self.motion_status_topic = rospy.get_param(
            "~motion_status_topic",
            "/mycobot320/aruco/motion_status",
        )
        self.control_status_topic = rospy.get_param(
            "~control_status_topic",
            "/mycobot320/aruco_control_status",
        )
        self.target_pose_topic = rospy.get_param(
            "~target_pose_topic",
            "/mycobot320/aruco_target_pose",
        )

        self.initial_pose_enabled = bool(
            rospy.get_param("~initial_pose_enabled", True)
        )
        self.initial_pose = (
            float(rospy.get_param("~initial_x", -0.026)),
            float(rospy.get_param("~initial_y", -0.166)),
            float(rospy.get_param("~initial_z", 0.308)),
            float(rospy.get_param("~initial_roll", 2.851)),
            float(rospy.get_param("~initial_pitch", -0.483)),
            float(rospy.get_param("~initial_yaw", -1.176)),
        )
        self.open_gripper_value = float(
            rospy.get_param("~open_gripper_value", 80.0)
        )
        self.gripper_open_settle_time = max(
            0.0,
            float(rospy.get_param("~gripper_open_settle_time", 2.0)),
        )

        self.offset_xyz = (
            float(rospy.get_param("~offset_x", 0.0)),
            float(rospy.get_param("~offset_y", 0.0)),
            float(rospy.get_param("~offset_z", 0.20)),
        )
        self.offset_rpy = (
            float(rospy.get_param("~offset_roll", math.pi)),
            float(rospy.get_param("~offset_pitch", 0.0)),
            float(rospy.get_param("~offset_yaw", 0.0)),
        )
        self.gripper_value = float(rospy.get_param("~gripper_value", -1.0))
        self.close_gripper_value = float(
            rospy.get_param("~close_gripper_value", 0.0)
        )
        self.retreat_distance = float(
            rospy.get_param("~retreat_distance", 0.08)
        )
        self.gripper_close_settle_time = max(
            0.0,
            float(rospy.get_param("~gripper_close_settle_time", 2.0)),
        )
        self.arm_goal_tolerance = float(
            rospy.get_param("~arm_goal_tolerance", 0.220)
        )

        self.startup_timeout = max(
            0.1,
            float(rospy.get_param("~startup_timeout", 60.0)),
        )
        self.tf_wait_timeout = max(
            0.1,
            float(rospy.get_param("~tf_wait_timeout", 30.0)),
        )
        self.tf_lookup_timeout = max(
            0.01,
            float(rospy.get_param("~tf_lookup_timeout", 0.25)),
        )
        self.max_tf_age = float(rospy.get_param("~max_tf_age", 1.0))
        self.stable_samples = max(
            1,
            int(rospy.get_param("~stable_samples", 5)),
        )
        self.sample_interval = max(
            0.02,
            float(rospy.get_param("~sample_interval", 0.10)),
        )
        self.max_position_jitter = max(
            0.0,
            float(rospy.get_param("~max_position_jitter", 0.008)),
        )
        self.max_orientation_jitter = max(
            0.0,
            float(rospy.get_param("~max_orientation_jitter", 0.10)),
        )
        self.motion_timeout = max(
            0.1,
            float(rospy.get_param("~motion_timeout", 60.0)),
        )
        self.expect_execution = bool(rospy.get_param("~expect_execution", True))
        self.wait_for_result = bool(rospy.get_param("~wait_for_result", True))

        if self.gripper_value >= 0.0 and not 0.0 <= self.gripper_value <= 100.0:
            raise ValueError("gripper_value must be 0-100, or negative to leave it unchanged.")
        if not 0.0 <= self.open_gripper_value <= 100.0:
            raise ValueError("open_gripper_value must be in 0-100.")
        if not 0.0 <= self.close_gripper_value <= 100.0:
            raise ValueError("close_gripper_value must be in 0-100.")
        if not math.isfinite(self.retreat_distance) or not 0.0 <= self.retreat_distance <= 0.30:
            raise ValueError("retreat_distance must be a finite value in 0-0.30 m.")
        if not math.isfinite(self.arm_goal_tolerance) or self.arm_goal_tolerance <= 0.0:
            raise ValueError("arm_goal_tolerance must be a positive finite value.")
        if not all(math.isfinite(value) for value in self.offset_xyz + self.offset_rpy):
            raise ValueError("ArUco xyz/rpy offsets must be finite.")
        if not all(math.isfinite(value) for value in self.initial_pose):
            raise ValueError("Initial xyz/rpy pose must be finite.")

        self.motion_condition = threading.Condition()
        self.motion_events = []
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.command_pub = rospy.Publisher(
            self.command_topic,
            Float64MultiArray,
            queue_size=1,
        )
        self.target_pose_pub = rospy.Publisher(
            self.target_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self.control_status_pub = rospy.Publisher(
            self.control_status_topic,
            String,
            queue_size=5,
            latch=True,
        )
        self.motion_status_sub = rospy.Subscriber(
            self.motion_status_topic,
            String,
            self.motion_status_callback,
            queue_size=50,
        )

        rospy.loginfo(
            "ArUco control target: id=%d frame=%s, base=%s",
            self.marker_id,
            self.marker_frame,
            self.base_frame,
        )
        rospy.loginfo(
            "Initial pose enabled=%s: xyz=[%.3f, %.3f, %.3f] "
            "rpy=[%.3f, %.3f, %.3f], open gripper=%.1f",
            self.initial_pose_enabled,
            self.initial_pose[0],
            self.initial_pose[1],
            self.initial_pose[2],
            self.initial_pose[3],
            self.initial_pose[4],
            self.initial_pose[5],
            self.open_gripper_value,
        )
        rospy.loginfo(
            "Marker-local target offset: xyz=[%.3f, %.3f, %.3f] "
            "rpy=[%.3f, %.3f, %.3f]",
            self.offset_xyz[0],
            self.offset_xyz[1],
            self.offset_xyz[2],
            self.offset_rpy[0],
            self.offset_rpy[1],
            self.offset_rpy[2],
        )
        rospy.loginfo(
            "Post-reach action: close gripper to %.1f, then retreat %.3f m "
            "toward the base origin within the marker XY plane after %.1f s settling.",
            self.close_gripper_value,
            self.retreat_distance,
            self.gripper_close_settle_time,
        )

    def publish_control_status(self, status):
        self.control_status_pub.publish(String(data=status))
        rospy.loginfo("ArUco control status: %s", status)

    def motion_status_callback(self, msg):
        with self.motion_condition:
            self.motion_events.append(msg.data)
            if len(self.motion_events) > 200:
                self.motion_events = self.motion_events[-200:]
            self.motion_condition.notify_all()

    def wait_for_planner(self):
        deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.command_pub.get_num_connections() == 0:
            if rospy.Time.now() >= deadline:
                rospy.logerr(
                    "No subscriber connected to command topic %s.",
                    self.command_topic,
                )
                return False
            rospy.loginfo_throttle(
                5.0,
                "Waiting for plan_to_pose_joint_state on %s...",
                self.command_topic,
            )
            rate.sleep()
        return not rospy.is_shutdown()

    def target_from_transform(self, transform, offset_xyz=None, offset_rpy=None):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        q_base_marker = normalize_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        target_offset_xyz = self.offset_xyz if offset_xyz is None else offset_xyz
        base_offset = rotate_vector(q_base_marker, target_offset_xyz)
        position = (
            translation.x + base_offset[0],
            translation.y + base_offset[1],
            translation.z + base_offset[2],
        )
        q_marker_target = quaternion_from_euler(
            *(self.offset_rpy if offset_rpy is None else offset_rpy)
        )
        orientation = quaternion_multiply(q_base_marker, q_marker_target)
        return position, orientation

    def transform_is_fresh(self, transform):
        if self.max_tf_age <= 0.0 or transform.header.stamp == rospy.Time(0):
            return True
        age = (rospy.Time.now() - transform.header.stamp).to_sec()
        if age < 0.0:
            return True
        if age <= self.max_tf_age:
            return True
        rospy.logwarn_throttle(
            2.0,
            "Ignoring stale %s TF (age %.3f s, limit %.3f s).",
            self.marker_frame,
            age,
            self.max_tf_age,
        )
        return False

    def wait_for_stable_target(self):
        deadline = rospy.Time.now() + rospy.Duration(self.tf_wait_timeout)
        previous = None
        stable_count = 0

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.marker_frame,
                    rospy.Time(0),
                    rospy.Duration(self.tf_lookup_timeout),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                stable_count = 0
                previous = None
                rospy.logwarn_throttle(
                    2.0,
                    "Waiting for TF %s -> %s: %s",
                    self.base_frame,
                    self.marker_frame,
                    exc,
                )
                rospy.sleep(self.sample_interval)
                continue

            if not self.transform_is_fresh(transform):
                stable_count = 0
                previous = None
                rospy.sleep(self.sample_interval)
                continue

            current = self.target_from_transform(transform)
            if previous is None:
                stable_count = 1
            else:
                position_delta = math.sqrt(
                    sum(
                        (current[0][index] - previous[0][index]) ** 2
                        for index in range(3)
                    )
                )
                orientation_delta = quaternion_angle(current[1], previous[1])
                if (
                    position_delta <= self.max_position_jitter
                    and orientation_delta <= self.max_orientation_jitter
                ):
                    stable_count += 1
                else:
                    stable_count = 1
                    rospy.logwarn_throttle(
                        2.0,
                        "ArUco TF is moving: delta=%.4f m / %.4f rad.",
                        position_delta,
                        orientation_delta,
                    )
            previous = current

            rospy.loginfo_throttle(
                1.0,
                "Stable ArUco TF samples: %d/%d",
                stable_count,
                self.stable_samples,
            )
            if stable_count >= self.stable_samples:
                return current, transform
            rospy.sleep(self.sample_interval)

        rospy.logerr(
            "Timed out waiting for stable TF %s -> %s.",
            self.base_frame,
            self.marker_frame,
        )
        return None

    def publish_target_pose(self, target):
        position, orientation = target
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]
        self.target_pose_pub.publish(pose)
        return pose

    def publish_motion_command(self, target):
        position, orientation = target
        roll, pitch, yaw = quaternion_to_euler(*orientation)
        command = Float64MultiArray()
        command.data = [
            position[0],
            position[1],
            position[2],
            roll,
            pitch,
            yaw,
            self.gripper_value,
            self.arm_goal_tolerance,
        ]
        with self.motion_condition:
            start_index = len(self.motion_events)
        self.command_pub.publish(command)
        rospy.loginfo("Published ArUco target command: %s", command.data)
        return start_index

    def publish_gripper_command(self, gripper_value, label):
        command = Float64MultiArray()
        command.data = [
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            gripper_value,
        ]
        with self.motion_condition:
            start_index = len(self.motion_events)
        self.command_pub.publish(command)
        rospy.loginfo("Published %s gripper command: value=%.1f", label, gripper_value)
        return start_index

    def initial_pose_target(self):
        return (
            self.initial_pose[:3],
            quaternion_from_euler(*self.initial_pose[3:]),
        )

    def retreat_target(self, marker_transform):
        approach_target = self.target_from_transform(marker_transform)
        if self.retreat_distance == 0.0:
            return approach_target

        rotation = marker_transform.transform.rotation
        q_base_marker = normalize_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        vector_to_base = tuple(-value for value in approach_target[0])
        vector_to_base_marker = rotate_vector(
            quaternion_conjugate(q_base_marker),
            vector_to_base,
        )
        planar_norm = math.hypot(
            vector_to_base_marker[0],
            vector_to_base_marker[1],
        )
        if planar_norm < 1e-6:
            raise ValueError(
                "Cannot determine a retreat direction in the marker XY plane."
            )

        retreat_dx = self.retreat_distance * vector_to_base_marker[0] / planar_norm
        retreat_dy = self.retreat_distance * vector_to_base_marker[1] / planar_norm
        retreat_offset = (
            self.offset_xyz[0] + retreat_dx,
            self.offset_xyz[1] + retreat_dy,
            self.offset_xyz[2],
        )
        rospy.loginfo(
            "Marker-local retreat delta: xyz=[%.4f, %.4f, 0.0000] m",
            retreat_dx,
            retreat_dy,
        )
        return self.target_from_transform(
            marker_transform,
            offset_xyz=retreat_offset,
        )

    def wait_for_motion_result(self, start_index):
        deadline = rospy.Time.now() + rospy.Duration(self.motion_timeout)
        success_state = "REACHED" if self.expect_execution else "PLANNED"
        active_seen = False

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.motion_condition:
                new_events = self.motion_events[start_index:]
                start_index = len(self.motion_events)
                for status in new_events:
                    if status in ACTIVE_MOTION_STATES:
                        active_seen = True
                    if status == success_state and active_seen:
                        rospy.loginfo("ArUco target motion completed with %s.", status)
                        return True
                    if status in FAILED_MOTION_STATES:
                        rospy.logerr("ArUco target motion failed with %s.", status)
                        return False
                self.motion_condition.wait(timeout=0.2)

        rospy.logerr("Timed out waiting for ArUco target motion result.")
        return False

    def run(self):
        self.publish_control_status("INITIALIZING")
        if not self.wait_for_planner():
            self.publish_control_status("PLANNER_TIMEOUT")
            return False

        if self.initial_pose_enabled:
            self.publish_control_status("OPENING_GRIPPER")
            open_start_index = self.publish_gripper_command(
                self.open_gripper_value,
                "initial open",
            )
            if not self.wait_for_motion_result(open_start_index):
                self.publish_control_status("GRIPPER_OPEN_FAILED")
                return False
            if self.expect_execution and self.gripper_open_settle_time > 0.0:
                self.publish_control_status("GRIPPER_OPENING_SETTLE")
                rospy.sleep(self.gripper_open_settle_time)
            self.publish_control_status("GRIPPER_OPENED")

            self.publish_control_status("MOVING_TO_INITIAL")
            initial_start_index = self.publish_motion_command(
                self.initial_pose_target()
            )
            if not self.wait_for_motion_result(initial_start_index):
                self.publish_control_status("INITIAL_MOTION_FAILED")
                return False
            self.publish_control_status("INITIAL_REACHED")

        self.publish_control_status("WAITING_FOR_TF")
        stable_result = self.wait_for_stable_target()
        if stable_result is None:
            self.publish_control_status("TF_TIMEOUT")
            return False
        target, marker_transform = stable_result

        pose = self.publish_target_pose(target)
        roll, pitch, yaw = quaternion_to_euler(
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        rospy.loginfo(
            "Target link6 pose in %s: xyz=[%.4f, %.4f, %.4f] "
            "rpy=[%.4f, %.4f, %.4f]",
            self.base_frame,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            roll,
            pitch,
            yaw,
        )
        self.publish_control_status("TARGET_READY")
        start_index = self.publish_motion_command(target)
        self.publish_control_status("COMMAND_SENT")

        if not self.wait_for_result:
            return True
        if not self.wait_for_motion_result(start_index):
            self.publish_control_status("MOTION_FAILED")
            return False

        self.publish_control_status("TARGET_REACHED")

        self.publish_control_status("CLOSING_GRIPPER")
        close_start_index = self.publish_gripper_command(
            self.close_gripper_value,
            "post-reach close",
        )
        if not self.wait_for_motion_result(close_start_index):
            self.publish_control_status("GRIPPER_CLOSE_FAILED")
            return False
        if self.expect_execution and self.gripper_close_settle_time > 0.0:
            self.publish_control_status("GRIPPER_SETTLING")
            rospy.sleep(self.gripper_close_settle_time)
        self.publish_control_status("GRIPPER_CLOSED")

        retreat_target = self.retreat_target(marker_transform)
        self.publish_target_pose(retreat_target)
        self.publish_control_status("RETREATING")
        retreat_start_index = self.publish_motion_command(retreat_target)
        if not self.wait_for_motion_result(retreat_start_index):
            self.publish_control_status("RETREAT_FAILED")
            return False

        self.publish_control_status("RETREAT_REACHED")
        self.publish_control_status("TASK_COMPLETE")
        return True


def main():
    rospy.init_node("aruco_pose_control")
    try:
        controller = ArucoPoseController()
        if not controller.run():
            sys.exit(1)
    except (ValueError, rospy.ROSException) as exc:
        rospy.logfatal("ArUco pose controller failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
