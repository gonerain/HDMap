#!/usr/bin/python3
import argparse
import bisect
import json
import math
import os
import pickle
import re
import signal
from collections import deque
from dataclasses import dataclass

import cv2
import genpy
import numpy as np
import rospy
from nav_msgs.msg import Path
from rosbag import Bag
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_from_matrix

import predict
from util import bri
from util import get_colors
from util import get_i_pcd_msg
from util import get_rgba_pcd_msg
from util import img2pcl
from util import pcl2image
from util import save_nppc


STOP_REQUESTED = False
STOP_REASON = None
STOP_AFTER_MAX_FRAMES = False


def request_stop(reason):
    global STOP_REQUESTED
    global STOP_REASON
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        STOP_REASON = str(reason)
        print(f"stop requested: {STOP_REASON}")
        if rospy.core.is_initialized() and not rospy.is_shutdown():
            rospy.signal_shutdown(STOP_REASON)


def handle_termination_signal(signum, _frame):
    signal_name = signal.Signals(signum).name
    request_stop(signal_name)


def request_max_frames_stop(max_frames):
    global STOP_AFTER_MAX_FRAMES
    if not STOP_AFTER_MAX_FRAMES:
        STOP_AFTER_MAX_FRAMES = True
        request_stop(f"reach max frames {max_frames}")


def cmkdir(path):
    path = path.split("/")
    cur = ""
    for p in path:
        try:
            cur = "/".join([cur, p])
            os.mkdir(cur[1:])
        except Exception:
            pass


def dms_to_deg(deg_token: str, minute_token: str, second_token: str) -> float:
    deg = float(deg_token)
    minute = float(minute_token)
    second = float(second_token)
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


@dataclass
class IePose:
    timestamp: float
    latitude_deg: float
    longitude_deg: float
    height_m: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float


def load_ie_poses(ie_path):
    if not os.path.exists(ie_path):
        raise FileNotFoundError(f"IE txt not found: {ie_path}")
    poses = []
    with open(ie_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or not re.match(r"^[0-9]", line):
            continue
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            timestamp = float(parts[0])
            lat_deg = dms_to_deg(parts[3], parts[4], parts[5])
            lon_deg = dms_to_deg(parts[6], parts[7], parts[8])
            height_m = float(parts[9])
            roll_deg = float(parts[-4])
            pitch_deg = float(parts[-3])
            heading_deg = float(parts[-2])
        except (ValueError, TypeError):
            continue
        poses.append(
            IePose(
                timestamp=timestamp,
                latitude_deg=lat_deg,
                longitude_deg=lon_deg,
                height_m=height_m,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                heading_deg=heading_deg,
            )
        )
    poses.sort(key=lambda x: x.timestamp)
    return poses


class IePoseProvider:
    def __init__(self, poses):
        self.poses = poses
        self.timestamps = [p.timestamp for p in poses]

    def get_interpolated(self, timestamp):
        if not self.poses:
            return None
        right = bisect.bisect_right(self.timestamps, float(timestamp))
        left = max(0, min(len(self.poses) - 1, right - 1))
        lp = self.poses[left]
        if left + 1 >= len(self.poses):
            return lp
        rp = self.poses[left + 1]
        dt = float(rp.timestamp - lp.timestamp)
        if dt <= 1e-9:
            return lp
        alpha = float(np.clip((timestamp - lp.timestamp) / dt, 0.0, 1.0))
        return IePose(
            timestamp=float(timestamp),
            latitude_deg=float(lp.latitude_deg + (rp.latitude_deg - lp.latitude_deg) * alpha),
            longitude_deg=float(lp.longitude_deg + (rp.longitude_deg - lp.longitude_deg) * alpha),
            height_m=float(lp.height_m + (rp.height_m - lp.height_m) * alpha),
            roll_deg=lerp_angle_deg(lp.roll_deg, rp.roll_deg, alpha),
            pitch_deg=lerp_angle_deg(lp.pitch_deg, rp.pitch_deg, alpha),
            heading_deg=lerp_angle_deg(lp.heading_deg, rp.heading_deg, alpha),
        )


def lerp_angle_deg(a_deg, b_deg, alpha):
    a = float(a_deg)
    b = float(b_deg)
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * float(alpha)


def deg2rad(v):
    return float(v) * math.pi / 180.0


def rotation_matrix_from_ie(roll_deg, pitch_deg, heading_deg):
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    neg_azimuth = -deg2rad(heading_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(neg_azimuth), math.sin(neg_azimuth)
    ry_passive_R = np.array([[cr, 0.0, -sr], [0.0, 1.0, 0.0], [sr, 0.0, cr]], dtype=np.float64)
    rx_passive_P = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]], dtype=np.float64)
    rz_passive_negA = np.array([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_rfu_from_enu = ry_passive_R @ rx_passive_P @ rz_passive_negA
    r_enu_from_rfu = r_rfu_from_enu.T
    r_rfu_from_flu = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return r_enu_from_rfu @ r_rfu_from_flu


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat_deg, lon_deg):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    return np.array(
        [[-sin_lon, cos_lon, 0.0], [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]],
        dtype=np.float64,
    )


def pose_enu_from_ie(ie_pose, ecef_origin, enu_from_ecef):
    ecef = geodetic_to_ecef(ie_pose.latitude_deg, ie_pose.longitude_deg, ie_pose.height_m)
    return enu_from_ecef @ (ecef - ecef_origin)


def ie_pose_to_quaternion_xyzw(ie_pose):
    r = rotation_matrix_from_ie(ie_pose.roll_deg, ie_pose.pitch_deg, ie_pose.heading_deg)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    q = quaternion_from_matrix(T)
    return np.asarray(q, dtype=np.float64)


# base<-lidar, copied from tracking project (FLU convention)
R_BASE_FROM_LIDAR = np.array([[0.9063, 0.0, 0.4226], [0.0, 1.0, 0.0], [-0.4226, 0.0, 0.9063]], dtype=np.float64)
T_BASE_FROM_LIDAR_M = np.array([0.0315, 0.0, 0.1314], dtype=np.float64)
R_BASE_FROM_SPAN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
T_BASE_FROM_SPAN_M = np.array([-0.2684, -0.0820, -0.1527], dtype=np.float64)
R_SPAN_FROM_BASE = R_BASE_FROM_SPAN.T
T_SPAN_FROM_BASE_M = -R_SPAN_FROM_BASE @ T_BASE_FROM_SPAN_M
R_SPAN_FROM_LIDAR = R_SPAN_FROM_BASE @ R_BASE_FROM_LIDAR
T_SPAN_FROM_LIDAR_M = R_SPAN_FROM_BASE @ T_BASE_FROM_LIDAR_M + T_SPAN_FROM_BASE_M
R_LIDAR_FROM_SPAN = R_SPAN_FROM_LIDAR.T
T_LIDAR_FROM_SPAN_M = -R_LIDAR_FROM_SPAN @ T_SPAN_FROM_LIDAR_M


def lidar_to_ie_body(points_lidar_xyz):
    pts = np.asarray(points_lidar_xyz, dtype=np.float64)
    p_span_frd = (R_SPAN_FROM_LIDAR @ pts.T).T + T_SPAN_FROM_LIDAR_M[None, :]
    out = np.empty_like(p_span_frd)
    out[:, 0] = p_span_frd[:, 0]
    out[:, 1] = -p_span_frd[:, 1]
    out[:, 2] = -p_span_frd[:, 2]
    return out


def ie_body_to_lidar(points_body_flu):
    pts = np.asarray(points_body_flu, dtype=np.float64)
    p_span_frd = np.empty_like(pts)
    p_span_frd[:, 0] = pts[:, 0]
    p_span_frd[:, 1] = -pts[:, 1]
    p_span_frd[:, 2] = -pts[:, 2]
    return (R_LIDAR_FROM_SPAN @ p_span_frd.T).T + T_LIDAR_FROM_SPAN_M[None, :]


def livox_msg_to_points_with_meta(msg):
    if getattr(msg, "_type", "") != "livox_ros_driver2/CustomMsg":
        raise RuntimeError(f"Unsupported lidar msg type: {getattr(msg, '_type', '')}, expected livox_ros_driver2/CustomMsg")
    point_count = len(msg.points)
    points = np.empty((point_count, 4), dtype=np.float64)
    offsets_s = np.empty((point_count,), dtype=np.float64)
    for i, p in enumerate(msg.points):
        points[i, 0] = float(p.x)
        points[i, 1] = float(p.y)
        points[i, 2] = float(p.z)
        points[i, 3] = float(getattr(p, "reflectivity", 0.0))
        offsets_s[i] = float(getattr(p, "offset_time", 0.0)) * 1e-9
    base_ts = float(getattr(msg, "timebase", 0)) * 1e-9
    if base_ts <= 0.0:
        base_ts = float(msg.header.stamp.to_sec())
    return {
        "base_timestamp": base_ts,
        "points_xyzi": points,
        "offsets_s": offsets_s,
    }


def decode_image_msg(msg):
    msg_type = getattr(msg, "_type", "")
    if msg_type == "sensor_msgs/CompressedImage":
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode CompressedImage")
        return frame
    if msg_type == "sensor_msgs/Image":
        h = int(msg.height)
        w = int(msg.width)
        enc = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if enc == "mono8":
            gray = data.reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if enc in {"bgr8", "rgb8"}:
            frame = data.reshape(h, w, 3)
            if enc == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame.copy()
    raise RuntimeError(f"Unsupported image msg type: {msg_type}")


def get_semantic_pcd(img, pcd_xyz, K, dismatrix, extrinsic, predictor):
    rimg = cv2.undistort(img, K, dismatrix)
    src = pcl2image(pcd_xyz, img.shape[1], img.shape[0], extrinsic)
    cimg = predictor(rimg)
    src[:, :, 2] = cimg
    sem_pcdata = img2pcl(src)
    if len(sem_pcdata) == 0:
        return np.array([]).reshape((0, 4)), cimg
    return sem_pcdata, cimg


def _ie_pose_to_enu_transform(ie_pose, ecef_origin, enu_from_ecef):
    r_enu_from_body = rotation_matrix_from_ie(ie_pose.roll_deg, ie_pose.pitch_deg, ie_pose.heading_deg)
    t_enu = pose_enu_from_ie(ie_pose, ecef_origin, enu_from_ecef)
    return r_enu_from_body, t_enu


def undistort_livox_points_to_base(points_xyzi, offsets_s, base_pose, ie_provider, ecef_origin, enu_from_ecef):
    if len(points_xyzi) == 0:
        return points_xyzi
    base_r_enu_from_body, base_t_enu = _ie_pose_to_enu_transform(base_pose, ecef_origin, enu_from_ecef)
    out = np.array(points_xyzi, dtype=np.float64, copy=True)
    for idx, offset_s in enumerate(offsets_s.tolist()):
        point_pose = ie_provider.get_interpolated(base_pose.timestamp + float(offset_s))
        if point_pose is None:
            continue
        point_r_enu_from_body, point_t_enu = _ie_pose_to_enu_transform(point_pose, ecef_origin, enu_from_ecef)
        point_body = lidar_to_ie_body(out[idx : idx + 1, :3])[0]
        point_enu = point_r_enu_from_body @ point_body + point_t_enu
        base_body = base_r_enu_from_body.T @ (point_enu - base_t_enu)
        out[idx, :3] = ie_body_to_lidar(base_body.reshape(1, 3))[0]
    return out


def find_bracketing_lidar(lidar_queue, t_img):
    if len(lidar_queue) < 2:
        return None
    l_next = None
    l_last = None
    for item in reversed(lidar_queue):
        if float(item["timestamp"]) < float(t_img):
            l_last = item
            break
        l_next = item
    if l_last is None or l_next is None:
        return None
    return l_last, l_next


def choose_nearest_lidar(lidar_queue, t_img, max_dt):
    pair = find_bracketing_lidar(lidar_queue, t_img)
    if pair is None:
        return None
    l_last, l_next = pair
    dt_last = abs(float(t_img) - float(l_last["timestamp"]))
    dt_next = abs(float(l_next["timestamp"]) - float(t_img))
    chosen = l_next if dt_next < dt_last else l_last
    chosen_dt = min(dt_last, dt_next)
    if max_dt is not None and chosen_dt > float(max_dt):
        return None
    return chosen


def transform_lidar_points_between_timestamps(points_src_lidar, src_pose, dst_pose, ecef_origin, enu_from_ecef):
    src_r_enu_from_body = rotation_matrix_from_ie(src_pose.roll_deg, src_pose.pitch_deg, src_pose.heading_deg)
    dst_r_enu_from_body = rotation_matrix_from_ie(dst_pose.roll_deg, dst_pose.pitch_deg, dst_pose.heading_deg)
    src_t_enu = pose_enu_from_ie(src_pose, ecef_origin, enu_from_ecef)
    dst_t_enu = pose_enu_from_ie(dst_pose, ecef_origin, enu_from_ecef)
    pts_body_src = lidar_to_ie_body(points_src_lidar[:, :3])
    pts_enu = (src_r_enu_from_body @ pts_body_src.T).T + src_t_enu[None, :]
    pts_body_dst = (dst_r_enu_from_body.T @ (pts_enu - dst_t_enu[None, :]).T).T
    out = np.array(points_src_lidar, dtype=np.float64, copy=True)
    out[:, :3] = ie_body_to_lidar(pts_body_dst)
    return out


def transform_semantic_lidar_to_world(sem_pcd_lidar_xyzc, dst_pose, ecef_origin, enu_from_ecef):
    if len(sem_pcd_lidar_xyzc) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    dst_r_enu_from_body = rotation_matrix_from_ie(dst_pose.roll_deg, dst_pose.pitch_deg, dst_pose.heading_deg)
    dst_t_enu = pose_enu_from_ie(dst_pose, ecef_origin, enu_from_ecef)
    pts_body = lidar_to_ie_body(sem_pcd_lidar_xyzc[:, :3])
    pts_enu = (dst_r_enu_from_body @ pts_body.T).T + dst_t_enu[None, :]
    out = np.zeros_like(sem_pcd_lidar_xyzc, dtype=np.float64)
    out[:, :3] = pts_enu
    out[:, 3] = sem_pcd_lidar_xyzc[:, 3]
    return out


def main():
    cmkdir("result/outdoor/originpics")
    cmkdir("result/outdoor/sempics")
    parser = argparse.ArgumentParser(description="Outdoor semantic point cloud builder for livox bag + IE txt")
    parser.add_argument("-c", "--config", default="config/outdoor_config_livox_ie.json")
    parser.add_argument("-b", "--bag", default=None)
    parser.add_argument("--ie-txt", default=None, help="Path to IE txt such as 0421-AM-2026.txt")
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument("--lidar-topic", default=None)
    parser.add_argument("-f", "--fastfoward", default=None, type=float)
    parser.add_argument("-d", "--duration", default=None, type=float)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--max-sync-dt", default=None, type=float, help="Max abs(camera_ts - lidar_ts) in seconds")
    parser.add_argument("--lidar-queue-size", default=None, type=int)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    bag_path = args.bag if args.bag is not None else config["bag_file"]
    ie_txt = args.ie_txt if args.ie_txt is not None else config.get("ie_txt", "data/outdoor/0421-AM-2026.txt")
    camera_topic = args.camera_topic if args.camera_topic is not None else config.get("camera_topic", "/front_camera/image/compressed")
    lidar_topic = args.lidar_topic if args.lidar_topic is not None else config.get("LiDAR_topic", "/lidar")
    start_offset = args.fastfoward if args.fastfoward is not None else config.get("start_time", 0.0)
    duration = args.duration if args.duration is not None else config.get("play_time", -1)
    max_sync_dt = args.max_sync_dt if args.max_sync_dt is not None else float(config.get("max_sync_dt", 0.12))
    lidar_queue_size = args.lidar_queue_size if args.lidar_queue_size is not None else int(config.get("lidar_queue_size", 300))
    road_class = config.get("road_class", 13)

    color_classes = get_colors(config["cmap"])
    K = np.asarray(config["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(config["extrinsic"], dtype=np.float64)
    dismatrix = np.asarray(config["distortion_matrix"], dtype=np.float64)
    colors = color_classes.astype("uint8")

    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)

    poses = load_ie_poses(ie_txt)
    if not poses:
        raise RuntimeError(f"No IE pose rows parsed from {ie_txt}")
    ie_provider = IePoseProvider(poses)
    ecef_origin = geodetic_to_ecef(poses[0].latitude_deg, poses[0].longitude_deg, poses[0].height_m)
    enu_from_ecef = ecef_to_enu_matrix(poses[0].latitude_deg, poses[0].longitude_deg)

    rospy.init_node("fix_distortion", anonymous=False, log_level=rospy.DEBUG, disable_signals=True)
    fixCloudPubHandle = rospy.Publisher("dedistortion_cloud", PointCloud2, queue_size=5)
    semanticCloudPubHandle = rospy.Publisher("SemanticCloud", PointCloud2, queue_size=5)
    roadCloudPubHandle = rospy.Publisher("RoadCloud", PointCloud2, queue_size=5)
    imgPubHandle = rospy.Publisher("Img", Image, queue_size=5)
    semimgPubHandle = rospy.Publisher("SemanticImg", Image, queue_size=5)
    groundTruthPubHandle = rospy.Publisher("ground_truth", Path, queue_size=0)
    _ = groundTruthPubHandle
    print("ros ready")

    predictor = getattr(predict, config["predict_func"])(config["model_config"], config["model_file"])
    print("torch ready")

    bag = None
    store_file = None
    pose_save = []
    road_save = []
    index = 0
    msg_total = 0
    msg_lidar = 0
    msg_camera = 0
    processed_with_lidar = 0

    lidar_queue = deque(maxlen=max(10, int(lidar_queue_size)))
    image_queue = deque()

    try:
        bag = Bag(bag_path)
        start = bag.get_start_time() + float(start_offset)
        start_t = genpy.Time.from_sec(start)
        end_t = None if float(duration) == -1 else genpy.Time.from_sec(start + float(duration))
        bagread = bag.read_messages(topics=[lidar_topic, camera_topic], start_time=start_t, end_time=end_t)
        print("bag ready")

        cmkdir(config["save_folder"] + "/originpics")
        cmkdir(config["save_folder"] + "/sempics")
        store_file = open(config["save_folder"] + "/outdoor.pkl", "wb")

        for topic, msg, stamp in bagread:
            if STOP_REQUESTED or STOP_AFTER_MAX_FRAMES or rospy.is_shutdown():
                break
            msg_total += 1
            ts = float(stamp.to_sec())
            if topic == lidar_topic:
                msg_lidar += 1
                try:
                    packet = livox_msg_to_points_with_meta(msg)
                except Exception as e:
                    print(f"skip lidar frame at {ts:.3f}: {e}")
                    continue
                base_ts = float(packet["base_timestamp"])
                lidar_pose = ie_provider.get_interpolated(base_ts)
                if lidar_pose is None:
                    continue
                undistorted_xyzi = undistort_livox_points_to_base(
                    packet["points_xyzi"],
                    packet["offsets_s"],
                    lidar_pose,
                    ie_provider,
                    ecef_origin,
                    enu_from_ecef,
                )
                lidar_queue.append({"timestamp": base_ts, "points_xyzi": undistorted_xyzi})
            elif topic == camera_topic:
                msg_camera += 1
                image_queue.append((ts, msg))
            else:
                continue

            while image_queue:
                if STOP_REQUESTED or STOP_AFTER_MAX_FRAMES or rospy.is_shutdown():
                    break
                img_ts, img_msg = image_queue[0]
                nearest = choose_nearest_lidar(lidar_queue, img_ts, max_sync_dt)
                if nearest is None:
                    break
                if args.max_frames is not None and index >= int(args.max_frames):
                    request_max_frames_stop(args.max_frames)
                    image_queue.clear()
                    break
                try:
                    img = decode_image_msg(img_msg)
                except Exception as e:
                    print(f"skip image frame at {img_ts:.3f}: {e}")
                    image_queue.popleft()
                    continue

                lidar_pose = ie_provider.get_interpolated(float(nearest["timestamp"]))
                image_pose = ie_provider.get_interpolated(float(img_ts))
                if lidar_pose is None or image_pose is None:
                    image_queue.popleft()
                    continue

                align_pcd = transform_lidar_points_between_timestamps(
                    nearest["points_xyzi"],
                    lidar_pose,
                    image_pose,
                    ecef_origin,
                    enu_from_ecef,
                )

                fixcloud = get_i_pcd_msg(align_pcd)
                fixcloud.header.frame_id = "lidar"
                fixCloudPubHandle.publish(fixcloud)

                index += 1
                processed_with_lidar += 1
                print(f"processing frame {index} img_ts={img_ts:.3f} lidar_ts={nearest['timestamp']:.3f}")

                q = ie_pose_to_quaternion_xyzw(image_pose)
                t_enu = pose_enu_from_ie(image_pose, ecef_origin, enu_from_ecef)
                pose_save.append(np.array([t_enu[0], t_enu[1], t_enu[2], q[0], q[1], q[2], q[3]], dtype=np.float64))

                imgPubHandle.publish(bri.cv2_to_imgmsg(img))
                cv2.imwrite(config["save_folder"] + "/originpics/%06d.png" % index, img)

                sem_pcd_lidar, semimg = get_semantic_pcd(img, align_pcd, K, dismatrix, extrinsic, predictor)
                cv2.imwrite(config["save_folder"] + "/sempics/%06d.png" % index, semimg)
                semimg_vis = colors[semimg.flatten()].reshape((*semimg.shape, 3))
                semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg_vis, "bgr8"))

                sem_world_pcd = transform_semantic_lidar_to_world(sem_pcd_lidar, image_pose, ecef_origin, enu_from_ecef)
                pickle.dump(sem_world_pcd, store_file)
                store_file.flush()

                if len(sem_world_pcd) != 0:
                    sem_msg = get_rgba_pcd_msg(sem_world_pcd)
                    sem_msg.header.frame_id = "world"
                    semanticCloudPubHandle.publish(sem_msg)
                    road_world_pcd = sem_world_pcd[sem_world_pcd[:, 3] == road_class]
                    if len(road_world_pcd) != 0:
                        road_save.append(road_world_pcd)
                        road_msg = get_rgba_pcd_msg(road_world_pcd)
                        road_msg.header.frame_id = "world"
                        roadCloudPubHandle.publish(road_msg)
                else:
                    print("semantic point publish skipped: empty semantic frame saved for alignment")
                image_queue.popleft()
            if STOP_REQUESTED or STOP_AFTER_MAX_FRAMES or rospy.is_shutdown():
                break
    except rospy.ROSInterruptException:
        request_stop("rospy.ROSInterruptException")
    except KeyboardInterrupt:
        request_stop("KeyboardInterrupt")
    finally:
        if store_file is not None:
            store_file.close()
        if bag is not None:
            bag.close()
        pose_array = np.stack(pose_save) if len(pose_save) != 0 else np.empty((0, 7))
        np.savetxt(config["save_folder"] + "/pose.csv", pose_array, delimiter=",")
        if len(road_save) != 0:
            save_nppc(np.vstack(road_save), config["save_folder"] + "/road.pcd")
        if STOP_REASON:
            print(f"exit: {STOP_REASON}")
        print(
            "summary: total_msgs=%d lidar=%d camera=%d processed_frames=%d aligned_frames=%d"
            % (msg_total, msg_lidar, msg_camera, index, processed_with_lidar)
        )
        if index == 0:
            print("warning: processed_frames is 0. Check topics/start_time/duration/max_sync_dt and IE txt range.")


if __name__ == "__main__":
    main()
