#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import threading

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray, String
from vision_msgs.msg import Detection3DArray

from grasp_ros.msg import GraspCandidateArray


TERMINAL_MOTION_STATES = {
    "REACHED",
    "PLAN_FAILED",
    "EXECUTION_FAILED",
    "REACHED_TIMEOUT",
    "STATE_TIMEOUT",
    "QUEUE_FULL",
}
FAILED_MOTION_STATES = TERMINAL_MOTION_STATES - {"REACHED"}
ACTIVE_MOTION_STATES = {
    "QUEUED",
    "ACCEPTED",
    "WAITING_FOR_STATE",
    "PLANNING",
    "EXECUTING",
}


def quaternion_to_euler(x, y, z, w):
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


def quaternion_angle_error(q1, q2):
    dot = abs(
        q1[0] * q2[0]
        + q1[1] * q2[1]
        + q1[2] * q2[2]
        + q1[3] * q2[3]
    )
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def normalize_quaternion(q):
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    return q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm


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


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class ObserveAndQueryObject(object):
    def __init__(self):
        self.command_topic = rospy.get_param(
            "~command_topic",
            "/mycobot320/target_pose_rpy",
        )
        self.motion_status_topic = rospy.get_param(
            "~motion_status_topic",
            "/mycobot320/motion_status",
        )
        self.task_status_topic = rospy.get_param(
            "~task_status_topic",
            "/mycobot320/task_status",
        )
        self.yolo_3d_result_topic = rospy.get_param(
            "~yolo_3d_result_topic",
            "/yolo_3d_result",
        )
        self.grasp_candidates_topic = rospy.get_param(
            "~grasp_candidates_topic",
            "/grasp_ros/grasp_candidates",
        )
        self.selected_frame_topic = rospy.get_param(
            "~selected_frame_topic",
            "/mycobot320/selected_grasp_frame",
        )
        self.selected_transform_topic = rospy.get_param(
            "~selected_transform_topic",
            "/mycobot320/selected_grasp_transform",
        )
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.grasp_frame_prefix = rospy.get_param(
            "~grasp_frame_prefix",
            "grasp_ros_link6_target",
        )
        self.observe_pose = self.read_pose_param(
            "~observe_pose",
            "observe_pose",
            100.0,
        )
        self.release_gripper_value = clamp(
            float(rospy.get_param("~release_gripper_value", 100.0)),
            0.0,
            100.0,
        )
        self.release_pose = self.read_pose_param(
            "~release_pose",
            "release_pose",
            self.release_gripper_value,
        )
        self.observe_pose[6] = clamp(self.observe_pose[6], 0.0, 100.0)
        self.release_pose[6] = clamp(self.release_pose[6], 0.0, 100.0)
        self.target_ids = self.parse_target_ids()
        self.startup_timeout = float(rospy.get_param("~startup_timeout", 60.0))
        self.motion_timeout = float(rospy.get_param("~motion_timeout", 60.0))
        self.release_arm_goal_tolerance = float(
            rospy.get_param("~release_arm_goal_tolerance", 0.250)
        )
        if (
            not math.isfinite(self.release_arm_goal_tolerance)
            or self.release_arm_goal_tolerance <= 0.0
        ):
            raise ValueError("release_arm_goal_tolerance must be a positive finite value.")
        self.detection_timeout = float(rospy.get_param("~detection_timeout", 0.0))
        self.grasp_timeout = float(rospy.get_param("~grasp_timeout", 60.0))
        self.transform_timeout = float(rospy.get_param("~transform_timeout", 0.5))
        self.target_max_age = float(rospy.get_param("~target_max_age", 2.0))
        self.scan_rate = max(0.2, float(rospy.get_param("~scan_rate", 2.0)))
        legacy_grasp_gripper_value = float(
            rospy.get_param("~grasp_gripper_value", 0.0)
        )
        self.approach_gripper_value = clamp(
            float(rospy.get_param("~approach_gripper_value", 100.0)),
            0.0,
            100.0,
        )
        self.post_reach_gripper_value = clamp(
            float(rospy.get_param("~post_reach_gripper_value", legacy_grasp_gripper_value)),
            0.0,
            100.0,
        )
        self.grasp_offset_frame = rospy.get_param("~grasp_offset_frame", "target").lower()
        if self.grasp_offset_frame not in ("target", "base"):
            raise ValueError("grasp_offset_frame must be 'target' or 'base'.")
        self.grasp_offset_xyz = (
            float(rospy.get_param("~grasp_offset_x", 0.0)),
            float(rospy.get_param("~grasp_offset_y", 0.0)),
            float(rospy.get_param("~grasp_offset_z", 0.0)),
        )
        self.grasp_offset_rpy = (
            float(rospy.get_param("~grasp_offset_roll", 0.0)),
            float(rospy.get_param("~grasp_offset_pitch", 0.0)),
            float(rospy.get_param("~grasp_offset_yaw", 0.0)),
        )
        self.max_grasp_cycles = int(rospy.get_param("~max_grasp_cycles", 0))
        self.retry_delay = max(0.0, float(rospy.get_param("~retry_delay", 0.5)))
        self.observe_link = rospy.get_param("~observe_link", "link6")
        self.observe_position_tolerance = float(
            rospy.get_param("~observe_position_tolerance", 0.08)
        )
        self.observe_orientation_tolerance = float(
            rospy.get_param("~observe_orientation_tolerance", 0.60)
        )

        self.motion_condition = threading.Condition()
        self.motion_events = []
        self.detection_condition = threading.Condition()
        self.latest_detections = []
        self.latest_detection_stamp = rospy.Time(0)
        self.latest_detection_seq = 0
        self.latest_detection_event = 0
        self.grasp_condition = threading.Condition()
        self.accept_grasp_candidates = False
        self.minimum_grasp_source_stamp = rospy.Time(0)
        self.minimum_grasp_source_seq = 0
        self.latest_grasp_candidates = []
        self.latest_grasp_stamp = rospy.Time(0)
        self.latest_grasp_event = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.command_pub = rospy.Publisher(
            self.command_topic,
            Float64MultiArray,
            queue_size=5,
        )
        self.task_status_pub = rospy.Publisher(
            self.task_status_topic,
            String,
            queue_size=10,
            latch=True,
        )
        self.selected_frame_pub = rospy.Publisher(
            self.selected_frame_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.selected_transform_pub = rospy.Publisher(
            self.selected_transform_topic,
            TransformStamped,
            queue_size=1,
            latch=True,
        )
        self.motion_status_sub = rospy.Subscriber(
            self.motion_status_topic,
            String,
            self.motion_status_callback,
            queue_size=50,
        )
        self.yolo_result_sub = rospy.Subscriber(
            self.yolo_3d_result_topic,
            Detection3DArray,
            self.yolo_result_callback,
            queue_size=3,
        )
        self.grasp_candidates_sub = rospy.Subscriber(
            self.grasp_candidates_topic,
            GraspCandidateArray,
            self.grasp_candidates_callback,
            queue_size=3,
        )

        self.publish_task_status("INITIALIZING")
        rospy.loginfo("Target COCO IDs: %s", self.target_ids if self.target_ids else "all")
        rospy.loginfo("YOLO 3D result topic: %s", self.yolo_3d_result_topic)
        rospy.loginfo("Grasp candidates topic: %s", self.grasp_candidates_topic)
        rospy.loginfo("Grasp TF prefix: %s", self.grasp_frame_prefix)
        rospy.loginfo(
            "Fixed gripper values: observe/return=%.1f approach=%.1f post_reach=%.1f release=%.1f.",
            self.observe_pose[6],
            self.approach_gripper_value,
            self.post_reach_gripper_value,
            self.release_gripper_value,
        )
        rospy.loginfo(
            "Grasp target fine tune in %s frame: xyz=[%.4f, %.4f, %.4f] rpy=[%.4f, %.4f, %.4f].",
            self.grasp_offset_frame,
            self.grasp_offset_xyz[0],
            self.grasp_offset_xyz[1],
            self.grasp_offset_xyz[2],
            self.grasp_offset_rpy[0],
            self.grasp_offset_rpy[1],
            self.grasp_offset_rpy[2],
        )
        rospy.loginfo(
            "Observation tolerance for %s -> %s: position=%.3f m orientation=%.3f rad",
            self.base_frame,
            self.observe_link,
            self.observe_position_tolerance,
            self.observe_orientation_tolerance,
        )

    def read_pose_param(self, param_name, label, default_gripper_value):
        values = [float(value) for value in rospy.get_param(param_name)]
        if len(values) not in (6, 7):
            raise ValueError(
                "%s must be [x, y, z, roll, pitch, yaw] "
                "or include a seventh gripper value." % label
            )
        if len(values) == 6:
            values.append(float(default_gripper_value))
        return values

    def parse_target_ids(self):
        target_id = int(rospy.get_param("~target_id", -1))
        value = rospy.get_param("~target_ids", [])

        ids = []
        if target_id >= 0:
            ids.append(target_id)

        if value is None or value == "":
            return ids
        if isinstance(value, str):
            parsed = yaml.safe_load(value)
            value = parsed if isinstance(parsed, list) else [parsed]
        if not isinstance(value, (list, tuple)):
            value = [value]

        for item in value:
            class_id = int(item)
            if class_id >= 0 and class_id not in ids:
                ids.append(class_id)
        return ids

    def publish_task_status(self, status):
        self.task_status_pub.publish(String(data=status))
        rospy.loginfo("Task status: %s", status)

    def motion_status_callback(self, msg):
        with self.motion_condition:
            self.motion_events.append(msg.data)
            if len(self.motion_events) > 200:
                self.motion_events = self.motion_events[-200:]
            self.motion_condition.notify_all()

    def yolo_result_callback(self, msg):
        candidates = self.extract_detection_candidates(msg)
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

        with self.detection_condition:
            self.latest_detections = candidates
            self.latest_detection_stamp = stamp
            self.latest_detection_seq = int(msg.header.seq)
            self.latest_detection_event += 1
            self.detection_condition.notify_all()

    def grasp_candidates_callback(self, msg):
        with self.grasp_condition:
            if not self.accept_grasp_candidates:
                return

            if not self.grasp_candidate_batch_valid(msg):
                return

            candidates = [
                candidate
                for candidate in msg.candidates
                if self.grasp_candidate_allowed(candidate)
            ]
            stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
            self.latest_grasp_candidates = candidates
            self.latest_grasp_stamp = stamp
            self.latest_grasp_event += 1
            self.grasp_condition.notify_all()

    def grasp_candidate_allowed(self, candidate):
        if not candidate.frame_id:
            return False
        if self.target_ids and int(candidate.class_id) not in self.target_ids:
            return False
        return True

    def grasp_candidate_batch_valid(self, msg):
        if not msg.candidates:
            return True

        if msg.header.stamp == rospy.Time(0):
            rospy.logwarn_throttle(2.0, "Ignoring grasp batch with a zero source stamp.")
            return False

        # msg.header.seq is the ROS publication sequence, not the grasp batch id.
        expected_batch = int(msg.candidates[0].batch_seq)
        expected_suffix = "_batch_%d" % expected_batch
        expected_source_seq = int(msg.candidates[0].source_seq)

        if (
            self.minimum_grasp_source_stamp != rospy.Time(0)
            and msg.header.stamp < self.minimum_grasp_source_stamp
        ):
            rospy.logwarn_throttle(
                2.0,
                "Ignoring grasp batch %d from %.9f; TARGET_FOUND was triggered at %.9f.",
                expected_batch,
                msg.header.stamp.to_sec(),
                self.minimum_grasp_source_stamp.to_sec(),
            )
            return False

        if (
            msg.header.stamp == self.minimum_grasp_source_stamp
            and self.minimum_grasp_source_seq != 0
            and expected_source_seq != self.minimum_grasp_source_seq
        ):
            rospy.logwarn_throttle(
                2.0,
                "Ignoring grasp batch %d: source seq=%d, TARGET_FOUND source seq=%d.",
                expected_batch,
                expected_source_seq,
                self.minimum_grasp_source_seq,
            )
            return False

        for candidate in msg.candidates:
            if int(candidate.batch_seq) != expected_batch:
                rospy.logwarn_throttle(
                    2.0,
                    "Ignoring inconsistent grasp batch: expected=%d candidate=%d.",
                    expected_batch,
                    int(candidate.batch_seq),
                )
                return False
            if candidate.source_stamp != msg.header.stamp:
                rospy.logwarn_throttle(
                    2.0,
                    "Ignoring grasp batch %d with inconsistent source stamps.",
                    expected_batch,
                )
                return False
            if int(candidate.source_seq) != expected_source_seq:
                rospy.logwarn_throttle(
                    2.0,
                    "Ignoring grasp batch %d with inconsistent source sequence numbers.",
                    expected_batch,
                )
                return False
            if not candidate.frame_id.endswith(expected_suffix):
                rospy.logwarn_throttle(
                    2.0,
                    "Ignoring grasp candidate whose TF frame does not match batch %d: %s.",
                    expected_batch,
                    candidate.frame_id,
                )
                return False
        return True

    def begin_grasp_candidate_acceptance(
        self,
        minimum_source_stamp=None,
        minimum_source_seq=0,
    ):
        with self.grasp_condition:
            self.accept_grasp_candidates = True
            self.minimum_grasp_source_stamp = (
                minimum_source_stamp
                if minimum_source_stamp is not None
                else rospy.Time(0)
            )
            self.minimum_grasp_source_seq = int(minimum_source_seq)
            self.latest_grasp_candidates = []
            self.latest_grasp_stamp = rospy.Time(0)
            self.latest_grasp_event = 0
            self.grasp_condition.notify_all()

    def stop_grasp_candidate_acceptance(self):
        with self.grasp_condition:
            self.accept_grasp_candidates = False
            self.minimum_grasp_source_stamp = rospy.Time(0)
            self.minimum_grasp_source_seq = 0
            self.latest_grasp_candidates = []
            self.latest_grasp_stamp = rospy.Time(0)
            self.grasp_condition.notify_all()

    def result_category_id(self, result):
        if hasattr(result, "id"):
            raw_id = result.id
        elif hasattr(result, "hypothesis") and hasattr(result.hypothesis, "class_id"):
            raw_id = result.hypothesis.class_id
        else:
            return None

        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return None

    def result_score(self, result):
        try:
            return float(result.score)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def detection_category(self, detection):
        best_id = None
        best_score = -1.0
        target_priority = {
            target_id: index for index, target_id in enumerate(self.target_ids)
        }

        for result in detection.results:
            category_id = self.result_category_id(result)
            if category_id is None:
                continue
            score = self.result_score(result)

            if self.target_ids:
                if category_id not in target_priority:
                    continue
                if best_id is None or target_priority[category_id] < target_priority[best_id]:
                    best_id = category_id
                    best_score = score
                elif category_id == best_id and score > best_score:
                    best_score = score
            elif score > best_score:
                best_id = category_id
                best_score = score

        if best_id is None:
            return None
        return best_id, best_score

    def extract_detection_candidates(self, msg):
        candidates = []
        class_instance_counts = {}
        target_priority = {
            target_id: index for index, target_id in enumerate(self.target_ids)
        }

        for detection in msg.detections:
            category = self.detection_category(detection)
            if category is None:
                continue

            category_id, score = category
            instance_index = class_instance_counts.get(category_id, 0)
            class_instance_counts[category_id] = instance_index + 1
            priority = (
                target_priority[category_id]
                if self.target_ids
                else category_id
            )
            candidates.append(
                {
                    "priority": priority,
                    "class_id": category_id,
                    "instance_index": instance_index,
                    "score": score,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["priority"],
                item["instance_index"],
                -item["score"],
            )
        )
        return candidates

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
            rate.sleep()
        return not rospy.is_shutdown()

    def publish_pose_command(self, pose_values, arm_goal_tolerance=None):
        msg = Float64MultiArray()
        msg.data = list(pose_values)
        if arm_goal_tolerance is not None:
            msg.data.append(float(arm_goal_tolerance))
        with self.motion_condition:
            start_index = len(self.motion_events)
        self.command_pub.publish(msg)
        rospy.loginfo("Published target pose command: %s", msg.data)
        return start_index

    def observe_pose_close_enough(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.observe_link,
                rospy.Time(0),
                rospy.Duration(self.transform_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                3.0,
                "Cannot check observation pose tolerance with %s -> %s: %s",
                self.base_frame,
                self.observe_link,
                exc,
            )
            return False

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        position_error = math.sqrt(
            (translation.x - self.observe_pose[0]) ** 2
            + (translation.y - self.observe_pose[1]) ** 2
            + (translation.z - self.observe_pose[2]) ** 2
        )
        actual_q = (rotation.x, rotation.y, rotation.z, rotation.w)
        target_q = quaternion_from_euler(
            self.observe_pose[3],
            self.observe_pose[4],
            self.observe_pose[5],
        )
        orientation_error = quaternion_angle_error(actual_q, target_q)

        close_enough = (
            position_error <= self.observe_position_tolerance
            and orientation_error <= self.observe_orientation_tolerance
        )
        rospy.loginfo_throttle(
            2.0,
            "Observation pose error: position=%.3f m orientation=%.3f rad "
            "(tolerance %.3f m / %.3f rad)",
            position_error,
            orientation_error,
            self.observe_position_tolerance,
            self.observe_orientation_tolerance,
        )
        return close_enough

    def wait_for_motion_result(self, start_index, label, close_enough=None):
        deadline = rospy.Time.now() + rospy.Duration(self.motion_timeout)
        active_seen = False

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.motion_condition:
                new_events = self.motion_events[start_index:]
                start_index = len(self.motion_events)

                for status in new_events:
                    if status in ACTIVE_MOTION_STATES:
                        active_seen = True
                    if status == "REACHED" and active_seen:
                        return True
                    if status in FAILED_MOTION_STATES:
                        if close_enough is not None and close_enough():
                            rospy.logwarn(
                                "%s motion reported %s, but TF is inside the relaxed tolerance.",
                                label,
                                status,
                            )
                            return True
                        rospy.logerr(
                            "%s motion failed with status: %s",
                            label,
                            status,
                        )
                        return False

                if active_seen and close_enough is not None and close_enough():
                    rospy.loginfo("%s accepted by relaxed TF tolerance.", label)
                    return True

                self.motion_condition.wait(timeout=0.2)

        if close_enough is not None and close_enough():
            rospy.logwarn(
                "%s motion timed out, but TF is inside the relaxed tolerance.",
                label,
            )
            return True

        rospy.logerr("Timed out waiting for %s motion.", label)
        return False

    def fresh_detection_batch(self):
        with self.detection_condition:
            candidates = list(self.latest_detections)
            stamp = self.latest_detection_stamp
            source_seq = self.latest_detection_seq

        if not candidates:
            return None
        if (
            self.target_max_age > 0.0
            and stamp != rospy.Time(0)
            and (rospy.Time.now() - stamp).to_sec() > self.target_max_age
        ):
            return None
        return {
            "candidates": candidates,
            "stamp": stamp,
            "source_seq": source_seq,
        }

    def fresh_grasp_candidates(self):
        with self.grasp_condition:
            candidates = list(self.latest_grasp_candidates)
            stamp = self.latest_grasp_stamp

        if not candidates:
            return []
        if (
            self.target_max_age > 0.0
            and stamp != rospy.Time(0)
            and (rospy.Time.now() - stamp).to_sec() > self.target_max_age
        ):
            return []
        return candidates

    def best_grasp_candidate(self, candidates):
        return max(candidates, key=lambda candidate: float(candidate.score))

    def wait_for_target_detection(self):
        deadline = None
        if self.detection_timeout > 0.0:
            deadline = rospy.Time.now() + rospy.Duration(self.detection_timeout)

        while not rospy.is_shutdown():
            batch = self.fresh_detection_batch()
            if batch is not None:
                return batch

            if deadline is not None and rospy.Time.now() >= deadline:
                rospy.logerr("Timed out waiting for target detections.")
                return None

            rospy.loginfo_throttle(
                5.0,
                "Waiting for target IDs %s on %s.",
                self.target_ids if self.target_ids else "all",
                self.yolo_3d_result_topic,
            )
            with self.detection_condition:
                self.detection_condition.wait(timeout=0.2)

        return None

    def lookup_grasp_candidate(self, candidate):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                candidate.frame_id,
                rospy.Time(0),
                rospy.Duration(self.transform_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                3.0,
                "Cannot transform %s to %s: %s",
                candidate.frame_id,
                self.base_frame,
                exc,
            )
            return None

        if (
            self.target_max_age > 0.0
            and transform.header.stamp != rospy.Time(0)
            and (rospy.Time.now() - transform.header.stamp).to_sec()
            > self.target_max_age
        ):
            rospy.logwarn_throttle(3.0, "Ignoring stale grasp TF %s.", candidate.frame_id)
            return None

        return {
            "class_id": int(candidate.class_id),
            "instance_index": int(candidate.instance_index),
            "batch_seq": int(candidate.batch_seq),
            "source_seq": int(candidate.source_seq),
            "source_stamp": candidate.source_stamp,
            "score": float(candidate.score),
            "width": float(candidate.width),
            "depth": float(candidate.depth),
            "frame": candidate.frame_id,
            "transform": transform,
            "gripper_value": self.approach_gripper_value,
        }

    def wait_for_grasp_target(self, start_accepting=True):
        if start_accepting:
            self.begin_grasp_candidate_acceptance()
        deadline = rospy.Time.now() + rospy.Duration(self.grasp_timeout)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            candidates = self.fresh_grasp_candidates()
            if candidates:
                best_candidate = self.best_grasp_candidate(candidates)
                result = self.lookup_grasp_candidate(best_candidate)
                if result is not None:
                    self.stop_grasp_candidate_acceptance()
                    rospy.loginfo(
                        "Locked grasp candidate batch=%d class=%d idx=%d score=%.3f width=%.3f m gripper=%.1f frame=%s (%d candidates in batch).",
                        result["batch_seq"],
                        result["class_id"],
                        result["instance_index"],
                        result["score"],
                        result["width"],
                        result["gripper_value"],
                        result["frame"],
                        len(candidates),
                    )
                    return result
            rospy.loginfo_throttle(
                5.0,
                "Waiting for fresh grasp candidates on %s.",
                self.grasp_candidates_topic,
            )
            with self.grasp_condition:
                self.grasp_condition.wait(timeout=0.2)

        self.stop_grasp_candidate_acceptance()
        rospy.logerr("Timed out waiting for a matching grasp target TF.")
        return None

    def apply_grasp_offset(self, position, quaternion):
        offset_q = quaternion_from_euler(
            self.grasp_offset_rpy[0],
            self.grasp_offset_rpy[1],
            self.grasp_offset_rpy[2],
        )

        if self.grasp_offset_frame == "target":
            dx, dy, dz = rotate_vector(quaternion, self.grasp_offset_xyz)
            corrected_position = (
                position[0] + dx,
                position[1] + dy,
                position[2] + dz,
            )
            corrected_quaternion = quaternion_multiply(quaternion, offset_q)
        else:
            corrected_position = (
                position[0] + self.grasp_offset_xyz[0],
                position[1] + self.grasp_offset_xyz[1],
                position[2] + self.grasp_offset_xyz[2],
            )
            corrected_quaternion = quaternion_multiply(offset_q, quaternion)

        return corrected_position, corrected_quaternion

    def transform_to_pose_command(self, transform, gripper_value):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        position = (translation.x, translation.y, translation.z)
        quaternion = normalize_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        position, quaternion = self.apply_grasp_offset(position, quaternion)
        roll, pitch, yaw = quaternion_to_euler(
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
        )
        return [
            position[0],
            position[1],
            position[2],
            roll,
            pitch,
            yaw,
            gripper_value,
        ]

    def gripper_only_command(self, gripper_value):
        return [
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            clamp(gripper_value, 0.0, 100.0),
        ]

    def move_to_pose(
        self,
        pose_values,
        label,
        close_enough=None,
        arm_goal_tolerance=None,
    ):
        start_index = self.publish_pose_command(
            pose_values,
            arm_goal_tolerance=arm_goal_tolerance,
        )
        return self.wait_for_motion_result(
            start_index,
            label,
            close_enough=close_enough,
        )

    def run(self):
        if not self.wait_for_planner():
            self.publish_task_status("STARTUP_FAILED")
            return

        self.publish_task_status("OPENING_GRIPPER")
        if not self.move_to_pose(
            self.gripper_only_command(self.observe_pose[6]),
            "initial gripper open",
        ):
            self.publish_task_status("GRIPPER_OPEN_FAILED")
            return
        self.publish_task_status("GRIPPER_OPENED")

        self.publish_task_status("MOVING_TO_OBSERVE")
        if not self.move_to_pose(
            self.observe_pose,
            "observation pose",
            close_enough=self.observe_pose_close_enough,
        ):
            self.publish_task_status("OBSERVE_FAILED")
            return
        self.publish_task_status("OBSERVE_REACHED")

        completed_cycles = 0
        while not rospy.is_shutdown():
            self.publish_task_status("OBSERVING")
            detection_batch = self.wait_for_target_detection()
            if detection_batch is None:
                self.publish_task_status("TARGET_NOT_FOUND")
                if self.retry_delay > 0.0:
                    rospy.sleep(self.retry_delay)
                continue

            detection_candidates = detection_batch["candidates"]
            selected_detection = detection_candidates[0]
            rospy.loginfo(
                "Detected %d target instances; first class_id=%d instance=%d score=%.3f",
                len(detection_candidates),
                selected_detection["class_id"],
                selected_detection["instance_index"],
                selected_detection["score"],
            )
            self.begin_grasp_candidate_acceptance(
                detection_batch["stamp"],
                detection_batch["source_seq"],
            )
            self.publish_task_status("TARGET_FOUND")

            grasp = self.wait_for_grasp_target(start_accepting=False)
            if grasp is None:
                self.publish_task_status("GRASP_TARGET_TIMEOUT")
                if self.retry_delay > 0.0:
                    rospy.sleep(self.retry_delay)
                continue

            self.selected_frame_pub.publish(String(data=grasp["frame"]))
            self.selected_transform_pub.publish(grasp["transform"])
            self.publish_task_status("GRASP_TARGET_FOUND")
            rospy.loginfo(
                "Selected grasp target class=%d instance=%d batch=%d score=%.3f width=%.3f m approach_gripper=%.1f frame=%s",
                grasp["class_id"],
                grasp["instance_index"],
                grasp["batch_seq"],
                grasp["score"],
                grasp["width"],
                self.approach_gripper_value,
                grasp["frame"],
            )

            self.publish_task_status("MOVING_TO_GRASP")
            grasp_pose = self.transform_to_pose_command(
                grasp["transform"],
                self.approach_gripper_value,
            )
            if not self.move_to_pose(grasp_pose, "grasp target"):
                self.publish_task_status("GRASP_FAILED")
                return

            self.publish_task_status("GRASP_REACHED")

            self.publish_task_status("CLOSING_GRIPPER")
            if not self.move_to_pose(
                self.gripper_only_command(self.post_reach_gripper_value),
                "post-reach gripper close",
            ):
                self.publish_task_status("GRIPPER_CLOSE_FAILED")
                return
            self.publish_task_status("GRIPPER_CLOSED")

            release_hold_pose = list(self.release_pose)
            release_hold_pose[6] = self.post_reach_gripper_value
            self.publish_task_status("MOVING_TO_RELEASE")
            if not self.move_to_pose(
                release_hold_pose,
                "release pose",
                arm_goal_tolerance=self.release_arm_goal_tolerance,
            ):
                self.publish_task_status("RELEASE_FAILED")
                return

            release_open_pose = list(self.release_pose)
            release_open_pose[6] = self.release_gripper_value
            self.publish_task_status("RELEASING")
            if not self.move_to_pose(
                release_open_pose,
                "release gripper",
                arm_goal_tolerance=self.release_arm_goal_tolerance,
            ):
                self.publish_task_status("RELEASE_FAILED")
                return
            self.publish_task_status("RELEASED")

            completed_cycles += 1
            if self.max_grasp_cycles > 0 and completed_cycles >= self.max_grasp_cycles:
                self.publish_task_status("TASK_COMPLETE")
                break

            self.publish_task_status("RETURNING_TO_OBSERVE")
            if not self.move_to_pose(
                self.observe_pose,
                "observation pose",
                close_enough=self.observe_pose_close_enough,
            ):
                self.publish_task_status("OBSERVE_FAILED")
                return
            self.publish_task_status("OBSERVE_REACHED")
            if self.retry_delay > 0.0:
                rospy.sleep(self.retry_delay)

        rospy.spin()


def main():
    rospy.init_node("observe_and_query_object")
    node = ObserveAndQueryObject()
    node.run()


if __name__ == "__main__":
    main()
