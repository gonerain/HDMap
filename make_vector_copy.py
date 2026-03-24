#!/usr/bin/python3
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
import sys
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
    
    def is_full(self):
        if self.cnt == -1:
            return False
        return len(self) >= self.cnt

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

    #road
    global roadpcd
    global dircpcd
    global road_savedpcd
    global window_road
    global dirc_window

    if args.trajectory:
        p = poses[index]
        rotation = pd.Series(p[3:7], index=['x', 'y', 'z', 'w'])
        br.sendTransform((p[0], p[1], p[2]), rotation, rospy.Time(time.time()), 'odom', 'world')
        if args.vector:
            # pole can be vectorized globally
            # process lane
            front_start = index - dirc_window
            last_start = index + dirc_window

            roadpcd = list(road_savedpcd[dirc_window:(dirc_window + window_road)])
            roads = np.vstack(roadpcd)
            dircpcd = {
                "front": {
                    "pcd": list(road_savedpcd[:window_road]),
                    "centerpoint": np.zeros(2, dtype=np.float32),
                    "pose": poses[front_start:(front_start + window_road)].copy(),
                    "points": np.vstack(road_savedpcd[:window_road]),
                },
                "last": {
                    "pcd": list(road_savedpcd[(2 * dirc_window):(2 * dirc_window + window_road)]),
                    "centerpoint": np.zeros(2, dtype=np.float32),
                    "pose": poses[last_start:(last_start + window_road)].copy(),
                    "points": np.vstack(road_savedpcd[(2 * dirc_window):(2 * dirc_window + window_road)]),
                },
            }

            current_center = roads[:, :2].mean(axis=0).astype(np.float32)
            dircpcd["front"]["centerpoint"] = dircpcd["front"]["points"][:, :2].mean(axis=0).astype(np.float32)
            dircpcd["last"]["centerpoint"] = dircpcd["last"]["points"][:, :2].mean(axis=0).astype(np.float32)

            all_xy = np.vstack((
                roads[:, :2],
                dircpcd["front"]["points"][:, :2],
                dircpcd["last"]["points"][:, :2],
                current_center[None, :],
                dircpcd["front"]["centerpoint"][None, :],
                dircpcd["last"]["centerpoint"][None, :],
            ))
            min_xy = all_xy.min(axis=0)
            max_xy = all_xy.max(axis=0)
            span = np.maximum(max_xy - min_xy, 1e-3)
            canvas_size = 1024
            margin = 60
            scale = float(min((canvas_size - 2 * margin) / span[0], (canvas_size - 2 * margin) / span[1]))

            def to_canvas(points_xy):
                pts = np.asarray(points_xy, dtype=np.float32)
                canvas_pts = np.empty((len(pts), 2), dtype=np.int32)
                canvas_pts[:, 0] = np.round((pts[:, 0] - min_xy[0]) * scale + margin).astype(np.int32)
                canvas_pts[:, 1] = np.round((max_xy[1] - pts[:, 1]) * scale + margin).astype(np.int32)
                return canvas_pts

            canvas = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)
            color_map = {
                "current": ((roads[:, :2]), (30, 30, 30)),
                "front": ((dircpcd["front"]["points"][:, :2]), (0, 170, 255)),
                "last": ((dircpcd["last"]["points"][:, :2]), (0, 200, 0)),
            }
            for _, (points_xy, color) in color_map.items():
                for px, py in to_canvas(points_xy):
                    cv2.circle(canvas, (int(px), int(py)), 1, color, -1)

            current_pt = tuple(to_canvas(current_center[None, :])[0])
            front_pt = tuple(to_canvas(dircpcd["front"]["centerpoint"][None, :])[0])
            last_pt = tuple(to_canvas(dircpcd["last"]["centerpoint"][None, :])[0])
            cv2.circle(canvas, current_pt, 8, (255, 255, 255), -1)
            cv2.arrowedLine(canvas, current_pt, front_pt, (0, 170, 255), 4, tipLength=0.12)
            cv2.arrowedLine(canvas, current_pt, last_pt, (0, 200, 0), 4, tipLength=0.12)
            cv2.putText(canvas, "current centroid", (current_pt[0] + 12, current_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
            cv2.putText(canvas, "front", (front_pt[0] + 12, front_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 220), 2)
            cv2.putText(canvas, "last", (last_pt[0] + 12, last_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
            cv2.imwrite("road_centroid_debug.png", canvas)
            sys.exit(0)

            
    index += 1


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
pairs_all = []
vec_world = []

#road
window_road = 10
dirc_window = 20
road_savedpcd=myqueue(2*dirc_window + window_road)

index = dirc_window
roadpcd = myqueue(window_road)
dircpcd = {
    "front": {
        "pcd": myqueue(window_road),
        "centerpoint": np.zeros(2, dtype=np.float32),
        "pose": None
    },
    "last": {
        "pcd": myqueue(window_road),
        "centerpoint": np.zeros(2, dtype=np.float32),
        "pose": None
    }
}



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
            road_savedpcd.append(sempcd[sempcd[:, 3] == config['road_class']])
            if road_savedpcd.is_full() != True:
                continue
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
    polemsg = get_rgba_pcd_msg(np.vstack(poles),color2int32((255,0,255,0)))
    vecPubHandle.publish(polemsg)


if args.save is not None:
    save_nppc(savepcd,args.save)
    lane = np.vstack(vectors)
    p = np.vstack(poles)
    v = np.vstack((lane,p))
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
