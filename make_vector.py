#!/usr/bin/python3
import argparse

from core.vector_common import RuntimeContext
from core.vector_common import save_nppc
from core.vector_registry import PROCESS_REGISTRY
from core.vector_registry import iterate_frames
from core.vector_registry import load_args_with_config


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorize semantic targets from saved semantic point clouds")
    parser.add_argument("-c", "--config", help="The config file path, recommand use this method to start the tool")
    parser.add_argument("-i", "--input", type=argparse.FileType("rb"))
    parser.add_argument("-m", "--mode", choices=["outdoor", "indoor"], help="Depend on the way to store the pickle file")
    parser.add_argument("-f", "--filters", default=None, nargs="+", type=int, help="Default to show all the classes, the meaning of each class refers to class.json")
    parser.add_argument("-s", "--save", default=None, help="Save to pcd file")
    parser.add_argument("-t", "--trajectory", default=None, help="Trajectory file, use to follow the camera")
    parser.add_argument("--semantic", default=None, help="Semantic photos folder")
    parser.add_argument("--origin", default=None, help="Origin photos folder")
    parser.add_argument("--vector", default=None, help="Enable vectorization", action="store_true")
    parser.add_argument("--process", choices=sorted(PROCESS_REGISTRY.keys()), default=None, help="Which vectorization process to run")
    parser.add_argument("--target-class", default=None, type=int, help="Override the class id consumed by the selected process")
    parser.add_argument("--output", default=None, help="Output JSON path for vector records")
    parser.add_argument("--max_index", "--max-index", dest="max_index", default=10000, type=int, help="Max logical index to process")
    parser.add_argument("--start_index", "--start-index", dest="start_index", default=None, type=int, help="Start processing from this logical frame index")
    return parser.parse_args()


def main():
    args = parse_args()
    args, config = load_args_with_config(args)
    runtime = RuntimeContext(args, config)
    process_cls = PROCESS_REGISTRY[args.process]
    process_obj = process_cls(args, config)

    savepcd = iterate_frames(runtime, process_obj, args)

    if args.vector:
        process_obj.finalize()

    if args.save is not None and len(savepcd) != 0:
        save_nppc(savepcd, args.save, runtime.color_classes)


if __name__ == "__main__":
    main()

"""
Legacy lane/pole vectorization script kept below for reference.
It used to execute at module import time and intercepted argparse before the
process registry entrypoint above could handle --process/--output/--max-index.
The active CLI is now the registry-based main() above.

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

    if args.trajectory:
        p = poses[index]
        rotation = pd.Series(p[3:7], index=['x', 'y', 'z', 'w'])
        br.sendTransform((p[0], p[1], p[2]), rotation, rospy.Time(time.time()), 'odom', 'world')
        if args.vector:
            # pole can be vectorized globally
            # process lane
            lanes = sempcd[sempcd[:, 3] == config['lane_class']]
            if len(lanepcd) < window:
                lanepcd.append(lanes)
            else:
                lanepcd.append(lanes)
                if index % step == 0:
                    lanes = np.vstack(lanepcd)
                    lanes = pcd_trans(lanes, p, rotation, True)
                    lanes = lanes[lanes[:, 1] < 8] #3 for parking lot, 8 for science park
                    #testPubHandle.publish(get_rgba_pcd_msg(pcd_trans(lanes,p,rotation)))
                    centers = get_lane_centers(lanes)
                    if len(centers) != 0:
                        centers = list(pcd_trans(centers,p,rotation))
                        if last_points:
                            pairs = {}
                            lines = []
                            for i in sum(last_points,[]):
                                d = 100000
                                pair = (d, None, None)
                                for j in centers:
                                    dis = np.linalg.norm(i - j)
                                    if dis > 1:
                                        continue
                                    if d != min(d, dis):
                                        d = dis
                                        pair = (d, i, j)
                                if pair[2] is None:
                                    continue
                                if tuple(pair[2]) in pairs:
                                    if d < pairs[tuple(pair[2])][0]:
                                        pairs[tuple(pair[2])] = pair
                                else:
                                    pairs[tuple(pair[2])] = pair
                            pairs_all.append(pairs)
                            for i in pairs:
                                lines.append(draw_line(*(pairs[i][1:])))
                            if len(lines) != 0:
                                lines = np.vstack(lines)
                                #lines = pcd_trans(lines,p,rotation)
                                vectors.append(lines)
                                vecmsg = get_rgba_pcd_msg(lines)
                                vecmsg.header.frame_id = 'world'
                                vecPubHandle.publish(vecmsg)
                        last_points.append(centers)
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
        main()
    except KeyboardInterrupt:
        print("interrupted by user")

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
"""
