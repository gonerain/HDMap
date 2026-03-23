#!/usr/bin/python3
import sys
import pickle
import rospy
from sklearn.cluster import DBSCAN
from util import get_rgba_pcd_msg
from sensor_msgs.msg import PointCloud2,Image
import json
import numpy as np
import pandas as pd
import argparse
from pclpy import pcl
import tf
from cv_bridge import CvBridge
import time
import glob
import cv2
import re
from tf import transformations
from predict import get_colors

height = {'pole': 5, 'lane':-1.1}

global sempcd
global args
global index
global poses
global br
global savepcd
global odom_trans
global last_points
global vectors
global lanepcd


class myqueue(list):
    def __init__(self, cnt=-1):
        self.cnt = cnt
        self.index = 0

    def append(self, obj):
        self.index+=1
        if len(self) >= self.cnt and self.cnt != -1:
            self.remove(self[0])
        super().append(obj)

    def is_empty(self):
        if len(self) == 0:
            return True
        else:
            return False

def color2int32(tup):
    return np.array([*tup[1:], 255]).astype(np.uint8).view('uint32')[0]


def class2color(cls,alpha = False):
    c = color_classes[cls]
    if not alpha:
        return np.array(c).astype(np.uint8)
    else:
        return np.array([*c, 255]).astype(np.uint8)

def save_nppc(nparr,fname):
    s = nparr.shape
    if s[1] == 4:#rgb
        tmp = pcl.PointCloud.PointXYZRGBA(nparr[:,:3],np.array([color_classes[int(i)] for i in nparr[:,3]]))
    else:
        tmp = pcl.PointCloud.PointXYZ(nparr)
    pcl.io.save(fname,tmp)
    return tmp

def draw_line(p1,p2):
    assert isinstance(p1,np.ndarray) or isinstance(p1,set)
    assert isinstance(p2,np.ndarray) or isinstance(p2,set)
    assert p1.shape == p2.shape
    if len(p1.shape) == 2 or p1.shape[0]== 2:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        line = np.stack((x,y),axis=1)
    else:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2+(p1[2]-p2[2])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        z = np.linspace(p1[2],p2[2],n)
        line = np.stack((x,y,z),axis=1)
    return line


def pcd_trans(pcd,dt,dr,inverse = False):
    length = len(pcd)
    if not isinstance(pcd,np.ndarray):
        pcd = np.array(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    if inverse:
        mat44 = np.matrix(mat44).I
    pcd[:3] = np.dot(mat44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd


def pointcloud_to_bev(points, resolution=0.2, x_range=None, y_range=None, z_range=None, min_points_per_cell=1):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points must be an Nx3 or wider array')
    if resolution <= 0:
        raise ValueError('resolution must be positive')
    if min_points_per_cell < 1:
        raise ValueError('min_points_per_cell must be >= 1')

    xyz = points[:, :3].astype(np.float32, copy=False)
    valid = np.isfinite(xyz).all(axis=1)
    if points.shape[1] > 3:
        cls = points[:, 3]
        valid &= np.isfinite(cls)
        cls = cls[valid].astype(np.int32, copy=False)
    else:
        cls = None
    xyz = xyz[valid]
    point_indices = np.flatnonzero(valid).astype(np.int32, copy=False)
    if len(xyz) == 0:
        empty_shape = (0, 0)
        return {
            'bev': np.zeros(empty_shape, dtype=bool),
            'count_map': np.zeros(empty_shape, dtype=np.uint16),
            'x_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'y_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'z_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'xyz_mean_map': np.zeros((0, 0, 3), dtype=np.float32),
            'z_min_map': np.zeros(empty_shape, dtype=np.float32),
            'z_max_map': np.zeros(empty_shape, dtype=np.float32),
            'z_median_map': np.zeros(empty_shape, dtype=np.float32),
            'z_var_map': np.zeros(empty_shape, dtype=np.float32),
            'z_std_map': np.zeros(empty_shape, dtype=np.float32),
            'major_class_map': np.full(empty_shape, -1, dtype=np.int32),
            'class_hist_map': {},
            'point_indices_map': {},
            'origin_xy': np.zeros(2, dtype=np.float32),
            'resolution': float(resolution),
        }

    if x_range is not None:
        keep = (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] < x_range[1])
        xyz = xyz[keep]
        point_indices = point_indices[keep]
        if cls is not None:
            cls = cls[keep]
    if y_range is not None:
        keep = (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] < y_range[1])
        xyz = xyz[keep]
        point_indices = point_indices[keep]
        if cls is not None:
            cls = cls[keep]
    if z_range is not None:
        keep = (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] < z_range[1])
        xyz = xyz[keep]
        point_indices = point_indices[keep]
        if cls is not None:
            cls = cls[keep]
    if len(xyz) == 0:
        empty_shape = (0, 0)
        return {
            'bev': np.zeros(empty_shape, dtype=bool),
            'count_map': np.zeros(empty_shape, dtype=np.uint16),
            'x_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'y_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'z_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'xyz_mean_map': np.zeros((0, 0, 3), dtype=np.float32),
            'z_min_map': np.zeros(empty_shape, dtype=np.float32),
            'z_max_map': np.zeros(empty_shape, dtype=np.float32),
            'z_median_map': np.zeros(empty_shape, dtype=np.float32),
            'z_var_map': np.zeros(empty_shape, dtype=np.float32),
            'z_std_map': np.zeros(empty_shape, dtype=np.float32),
            'major_class_map': np.full(empty_shape, -1, dtype=np.int32),
            'class_hist_map': {},
            'point_indices_map': {},
            'origin_xy': np.zeros(2, dtype=np.float32),
            'resolution': float(resolution),
        }

    origin_xy = np.array([
        x_range[0] if x_range is not None else xyz[:, 0].min(),
        y_range[0] if y_range is not None else xyz[:, 1].min(),
    ], dtype=np.float32)
    max_xy = np.array([
        x_range[1] if x_range is not None else xyz[:, 0].max(),
        y_range[1] if y_range is not None else xyz[:, 1].max(),
    ], dtype=np.float32)

    span = np.maximum(max_xy - origin_xy, resolution)
    width = int(np.ceil(span[0] / resolution))
    height = int(np.ceil(span[1] / resolution))
    ij = np.floor((xyz[:, :2] - origin_xy) / resolution).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)

    count_map = np.zeros((height, width), dtype=np.uint16)
    np.add.at(count_map, (ij[:, 1], ij[:, 0]), 1)

    x_sum_map = np.zeros((height, width), dtype=np.float64)
    y_sum_map = np.zeros((height, width), dtype=np.float64)
    z_sum_map = np.zeros((height, width), dtype=np.float64)
    np.add.at(x_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 0])
    np.add.at(y_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 1])
    np.add.at(z_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 2])

    x_mean_map = np.divide(x_sum_map, count_map, out=np.zeros_like(x_sum_map, dtype=np.float64), where=count_map > 0).astype(np.float32)
    y_mean_map = np.divide(y_sum_map, count_map, out=np.zeros_like(y_sum_map, dtype=np.float64), where=count_map > 0).astype(np.float32)
    z_mean_map = np.divide(z_sum_map, count_map, out=np.zeros_like(z_sum_map, dtype=np.float64), where=count_map > 0).astype(np.float32)

    z_min_map = np.full((height, width), np.inf, dtype=np.float32)
    z_max_map = np.full((height, width), -np.inf, dtype=np.float32)
    np.minimum.at(z_min_map, (ij[:, 1], ij[:, 0]), xyz[:, 2])
    np.maximum.at(z_max_map, (ij[:, 1], ij[:, 0]), xyz[:, 2])
    z_min_map[~np.isfinite(z_min_map)] = 0.0
    z_max_map[~np.isfinite(z_max_map)] = 0.0

    z_sq_sum_map = np.zeros((height, width), dtype=np.float64)
    np.add.at(z_sq_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 2] * xyz[:, 2])
    z_var_map = np.divide(z_sq_sum_map, count_map, out=np.zeros_like(z_sq_sum_map, dtype=np.float64), where=count_map > 0) - z_mean_map.astype(np.float64) ** 2
    z_var_map = np.maximum(z_var_map, 0.0).astype(np.float32)
    z_std_map = np.sqrt(z_var_map, dtype=np.float32)

    xyz_mean_map = np.stack((x_mean_map, y_mean_map, z_mean_map), axis=-1)
    bev = count_map >= min_points_per_cell

    z_median_map = np.zeros((height, width), dtype=np.float32)
    point_indices_map = {}
    class_hist_map = {}
    major_class_map = np.full((height, width), -1, dtype=np.int32)

    flat_ids = ij[:, 1] * width + ij[:, 0]
    unique_ids = np.unique(flat_ids)
    for flat_id in unique_ids:
        mask = flat_ids == flat_id
        row = int(flat_id // width)
        col = int(flat_id % width)
        z_median_map[row, col] = np.median(xyz[mask, 2]).astype(np.float32)
        point_indices_map[(row, col)] = point_indices[mask].copy()
        if cls is not None:
            classes, hist = np.unique(cls[mask], return_counts=True)
            classes = classes.astype(np.int32, copy=False)
            hist = hist.astype(np.int32, copy=False)
            class_hist_map[(row, col)] = {
                int(c): int(h) for c, h in zip(classes, hist)
            }
            major_class_map[row, col] = int(classes[np.argmax(hist)])

    return {
        'bev': bev,
        'count_map': count_map,
        'x_mean_map': x_mean_map,
        'y_mean_map': y_mean_map,
        'z_mean_map': z_mean_map,
        'xyz_mean_map': xyz_mean_map,
        'z_min_map': z_min_map,
        'z_max_map': z_max_map,
        'z_median_map': z_median_map,
        'z_var_map': z_var_map,
        'z_std_map': z_std_map,
        'major_class_map': major_class_map,
        'class_hist_map': class_hist_map,
        'point_indices_map': point_indices_map,
        'origin_xy': origin_xy,
        'resolution': float(resolution),
    }


def morph_open(binary_img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(binary_img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def flood_fill_holes(binary_img):
    filled = binary_img.copy()
    height, width = filled.shape[:2]
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(binary_img, holes)


def keep_largest_connected_component(binary_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels <= 1:
        return binary_img.copy()
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output = np.zeros_like(binary_img)
    output[labels == largest_label] = 255
    return output


def stack_nonempty(items, min_cols=4):
    arrays = []
    for item in items:
        arr = np.asarray(item)
        if arr.ndim != 2:
            continue
        if arr.shape[1] < min_cols or len(arr) == 0:
            continue
        arrays.append(arr)
    if not arrays:
        return np.zeros((0, min_cols), dtype=np.float32)
    return np.vstack(arrays)


def bev_mask_to_points(bev_data, mask, z_value=None):
    if mask is None or mask.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rows, cols = np.nonzero(mask > 0)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    resolution = float(bev_data['resolution'])
    origin_xy = np.asarray(bev_data['origin_xy'], dtype=np.float32)
    xy = np.stack((
        origin_xy[0] + (cols.astype(np.float32) + 0.5) * resolution,
        origin_xy[1] + (rows.astype(np.float32) + 0.5) * resolution,
    ), axis=1)
    if z_value is None and 'z_mean_map' in bev_data:
        z = bev_data['z_mean_map'][rows, cols].astype(np.float32, copy=False).reshape(-1, 1)
    else:
        z = np.full((len(xy), 1), 0.0 if z_value is None else z_value, dtype=np.float32)
    return np.hstack((xy, z))


def merge_road_masks(mask_entries, resolution=0.2, min_votes=1):
    occupied_points = []
    for entry in mask_entries:
        occupied = bev_mask_to_points(entry['bev'], entry['mask'])
        if len(occupied) != 0:
            occupied_points.append(occupied)

    if not occupied_points:
        empty_shape = (0, 0)
        empty_bev = {
            'bev': np.zeros(empty_shape, dtype=bool),
            'count_map': np.zeros(empty_shape, dtype=np.uint16),
            'x_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'y_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'z_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'xyz_mean_map': np.zeros((0, 0, 3), dtype=np.float32),
            'z_min_map': np.zeros(empty_shape, dtype=np.float32),
            'z_max_map': np.zeros(empty_shape, dtype=np.float32),
            'z_median_map': np.zeros(empty_shape, dtype=np.float32),
            'z_var_map': np.zeros(empty_shape, dtype=np.float32),
            'z_std_map': np.zeros(empty_shape, dtype=np.float32),
            'major_class_map': np.full(empty_shape, -1, dtype=np.int32),
            'class_hist_map': {},
            'point_indices_map': {},
            'origin_xy': np.zeros(2, dtype=np.float32),
            'resolution': float(resolution),
        }
        return empty_bev, np.zeros(empty_shape, dtype=np.uint8)

    occupied_points = np.vstack(occupied_points)
    merged_bev = pointcloud_to_bev(
        occupied_points,
        resolution=resolution,
        min_points_per_cell=max(1, int(min_votes)),
    )
    merged_mask = merged_bev['bev'].astype(np.uint8) * 255
    if merged_mask.size != 0:
        merged_mask = morph_close(merged_mask, kernel_size=3, iterations=1)
        merged_mask = keep_largest_connected_component(merged_mask)
    return merged_bev, merged_mask


def filter_points_by_bev_mask(points, bev_data, mask):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return points
    if mask is None or mask.size == 0:
        return points[:0]

    height, width = mask.shape[:2]
    resolution = float(bev_data['resolution'])
    origin_xy = np.asarray(bev_data['origin_xy'], dtype=np.float32)

    ij = np.floor((points[:, :2] - origin_xy) / resolution).astype(np.int32)
    inside = (
        (ij[:, 0] >= 0) & (ij[:, 0] < width) &
        (ij[:, 1] >= 0) & (ij[:, 1] < height)
    )
    keep = np.zeros(len(points), dtype=bool)
    if np.any(inside):
        mask_values = mask[ij[inside, 1], ij[inside, 0]] > 0
        keep[np.flatnonzero(inside)] = mask_values
    return points[keep]


def get_lane_centers(pcd):
    centers = []
    if len(pcd) == 0:
        return centers
    dbs.fit(pcd)
    labels = dbs.fit_predict(pcd)  # label
    cluster = list(set(labels))
    n = len(cluster)
    for i in cluster:
        if n <= 0:
            continue
        c = pcd[labels == i]  # each cluster
        if abs(c[:,0].max()-c[:,0].min()) > 0.3:
            if c[:,0].mean() < 0:
                center = np.array((c[:, 0].max()-0.2, c[:, 1].mean(), c[:,2].mean()))#-2.3))
            else:
                center = np.array((c[:, 0].min()+0.2, c[:, 1].mean(), c[:,2].mean()))#-2.3))
        else:
            center = np.array((c[:,0].mean(),c[:,1].mean(),-2.3))
        centers.append(center)
    return centers

def get_pole_centers(pcd):
    centers = []
    if len(pcd) == 0:
        return centers
    pole_dbs.fit(pcd)
    labels = pole_dbs.fit_predict(pcd[:,:2])  # label
    cluster = list(set(labels))
    n = len(cluster)
    for i in cluster:
        if n <= 0:
            continue
        c = pcd[labels == i]  # each cluster
        center = np.array((c[:,0].mean(),c[:,1].mean(),c[:,2].max()))
        centers.append(center)
    return centers

def process():
    global sempcd
    global args
    global index
    global poses
    global br
    global last_points
    global vectors
    global lanepcd
    global roadpcd
    global road_masks

    if args.trajectory:
        p = poses[index]
        rotation = pd.Series(p[3:7], index=['x', 'y', 'z', 'w'])
        br.sendTransform((p[0], p[1], p[2]), rotation, rospy.Time(time.time()), 'odom', 'world')
        if args.vector:
            roads = sempcd[sempcd[:, 3] == config['road_class']]
            roadpcd.append(roads)
            if len(roadpcd) >= window:
                pcd_all = stack_nonempty(roadpcd)
                if len(pcd_all) == 0:
                    index += 1
                    return
                bev_data = pointcloud_to_bev(pcd_all, resolution=0.2)
                bev_img = bev_data['bev'].astype(np.uint8) * 255
                bev_img = morph_open(bev_img, kernel_size=1, iterations=1)
                bev_img = morph_close(bev_img, kernel_size=5, iterations=1)
                bev_img = flood_fill_holes(bev_img)
                road_mask = keep_largest_connected_component(bev_img)
                road_masks.append({
                    "pose": p,
                    "bev": bev_data,
                    "mask": road_mask
                })
                index += 1

                if len(road_masks) >= window:
                    print("len road_mask full")
                    sys.exit()
                else:
                    return


    if args.filters:
        sempcd = sempcd[np.in1d(sempcd[:, 3], args.filters)]
    sem_msg = get_rgba_pcd_msg(sempcd)
    sem_msg.header.frame_id = 'world'
    semanticCloudPubHandle.publish(sem_msg)

    if args.semantic and index < len(simgs):
        simg = cv2.imread(simgs[index],0)
        semimg = colors[simg.flatten()].reshape((*simg.shape,3))
        semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg, 'bgr8'))
    if args.origin and index < len(imgs):
        imgPubHandle.publish(bri.cv2_to_imgmsg(cv2.imread(imgs[index]), 'bgr8'))


# parse arguments
parser = argparse.ArgumentParser(description='Rebuild semantic point cloud')
parser.add_argument('-c','--config',help='The config file path, recommand use this method to start the tool')
parser.add_argument('-i','--input',type=argparse.FileType('rb'))
parser.add_argument('-m','--mode',choices=['outdoor','indoor'],help="Depend on the way to store the pickle file")
parser.add_argument('-f','--filters', default=None,nargs='+',type=int,help='Default to show all the classes, the meaning of each class refers to class.json')
parser.add_argument('-s','--save',default=None,help='Save to pcd file')
parser.add_argument('-t','--trajectory',default=None,help='Trajectory file, use to follow the camera')
parser.add_argument('--semantic',default=None,help='Semantic photos folder')
parser.add_argument('--origin',default=None,help='Origin photos folder')
parser.add_argument('--vector',default=None,help='Do the vectorization, only available when filters are accepted',action='store_true')
args = parser.parse_args()

if args.config:
    with open(args.config,'r') as f:
        config = json.load(f)

args.input = (args.input or open(config['save_folder']+(config['mode'] == 'indoor' and '/indoor.pkl' or '/outdoor.pkl'),'rb'))
args.mode = (args.mode or config['mode'])
args.trajectory = (args.trajectory or config['save_folder']+'/pose.csv')
args.save = (args.save or config['save_folder']+'/result.pcd')
args.semantic = (args.semantic or config['save_folder']+'/sempics')
args.origin = (args.origin or config['save_folder']+'/originpics')
args.vector = (args.vector) or config['vector']
# init variables

window = 10
step = 1

# start ros
rospy.init_node('fix_distortion', anonymous=False, log_level=rospy.DEBUG)
semanticCloudPubHandle = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=5)
vecPubHandle = rospy.Publisher('VectorCloud', PointCloud2, queue_size=5)
testPubHandle = rospy.Publisher('TestCloud', PointCloud2, queue_size=5)
semimgPubHandle = rospy.Publisher('SemanticImg',Image,queue_size = 5)
imgPubHandle = rospy.Publisher('Img',Image,queue_size = 5)

color_classes = get_colors(config['cmap'])
savepcd = []
vectors = []
bri = CvBridge()
index = 0
br = tf.TransformBroadcaster()
dbs = DBSCAN(eps = 1,min_samples=5,n_jobs=24)
pole_dbs = DBSCAN(eps = 0.3,min_samples=50,n_jobs=24)
#dbs = DBSCAN()
last_points = myqueue(1)
lanepcd = myqueue(window)
polepcd = myqueue(window)
roadpcd = myqueue(window)
pairs_all = []
vec_world = []

road_masks = myqueue(window)


if args.semantic:
    simgs = glob.glob(args.semantic+'/*')
    simgs.sort()
    #simgs.sort(key = lambda x:int(re.findall('[0-9]{4,5}',x)[0]))
    colors = color_classes.astype('uint8')

if args.origin:
    imgs = glob.glob(args.origin+'/*')
    imgs.sort()
    #imgs.sort(key = lambda x:int(re.findall('[0-9]{4,5}',x)[0]))

if args.trajectory:
    poses = np.loadtxt(args.trajectory, delimiter=',')


if args.mode == 'indoor':
    sempcds = pickle.load(args.input)
    for sempcd in sempcds:
        process()
    savepcd = np.concatenate(sempcds)
elif args.mode == 'outdoor':
    try:
        while True:
            sempcd = pickle.load(args.input)
            savepcd.append(sempcd)
            process()
            #print(index)
    except EOFError:
        print('done')
        savepcd = np.concatenate(savepcd)

if args.vector:
    poles = savepcd[np.in1d(savepcd[:, 3], [config['pole_class']])]
    poles = poles[poles[:,2]>-2]
    poles = poles[poles[:,2]<8]
    pole_centers = get_pole_centers(poles[:, :3])
    poles = []
    for pole in pole_centers:
        poles.append(draw_line(pole,np.array([pole[0],pole[1],-2.3])))
    if len(poles) != 0:
        polemsg = get_rgba_pcd_msg(np.vstack(poles),color2int32((255,0,255,0)))
        vecPubHandle.publish(polemsg)


if args.save is not None:
    save_nppc(savepcd,args.save)
    vector_parts = []
    if len(vectors) != 0:
        vector_parts.append(np.vstack(vectors))
    if args.vector and len(poles) != 0:
        vector_parts.append(np.vstack(poles))
    if len(vector_parts) != 0:
        v = np.vstack(vector_parts)
        save_nppc(v,'/'.join(args.save.split('/')[:-1])+'/vector.pcd')



def filter():
    srcpcd = './worldSemanticCloud.pcd'
    wsc = pcl.PointCloud.PointXYZRGBA()
    fpc = pcl.PointCloud.PointXYZRGBA()
    pcl.io.loadPCDFile(srcpcd,wsc)
    f = pcl.filters.RadiusOutlierRemoval.PointXYZRGBA()
    f.setInputCloud(wsc)
    f.setRadiusSearch(0.1)
    f.setMinNeighborsInRadius(10)
    f.filter(fpc)
