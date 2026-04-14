import json
import pickle

import numpy as np

from core.crosswalk_process import CrosswalkProcess
from core.road_process import RoadEdgeProcess
from core.road_process_v2 import RoadEdgeProcessV2
# from core.sidewalk_roadfused_process import SidewalkRoadFusedProcess  # removed
from core.sidewalk_process import SidewalkEdgeProcess


PROCESS_REGISTRY = {
    "crosswalk": CrosswalkProcess,
    "road": RoadEdgeProcess,
    "road_v2": RoadEdgeProcessV2,
    "sidewalk": SidewalkEdgeProcess,
    # "sidewalk_roadfused": SidewalkRoadFusedProcess,  # removed
}


def load_args_with_config(args):
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("--config is required")

    args.input = args.input or open(config["save_folder"] + ("/indoor.pkl" if config["mode"] == "indoor" else "/outdoor.pkl"), "rb")
    args.mode = args.mode or config["mode"]
    args.filters = args.filters or config.get("filters")
    args.trajectory = args.trajectory or config["save_folder"] + "/pose.csv"
    args.save = args.save or config["save_folder"] + "/result.pcd"
    args.semantic = args.semantic or config["save_folder"] + "/sempics"
    args.origin = args.origin or config["save_folder"] + "/originpics"
    args.vector = args.vector or config["vector"]
    args.process = args.process or config.get("vector_process", "road")
    if args.max_index == 10000:
        args.max_index = config.get("max_index", args.max_index)
    if args.start_index is None:
        args.start_index = config.get("start_index", 0)
    return args, config


def iterate_frames(runtime, process_obj, args):
    start_index = max(int(args.start_index or 0), 0)

    if args.mode == "indoor":
        savepcd = []
        try:
            sempcds = pickle.load(args.input)
            for frame_index, sempcd in enumerate(sempcds):
                savepcd.append(sempcd)
                runtime.publish_pose(frame_index)
                runtime.publish_frame(frame_index, sempcd)
            return np.concatenate(sempcds) if len(sempcds) != 0 else np.zeros((0, 4), dtype=np.float32)
        except KeyboardInterrupt:
            print("interrupted by user")
            return np.zeros((0, 4), dtype=np.float32)

    latest_frame_index = -1
    try:
        while True:
            sempcd = pickle.load(args.input)
            latest_frame_index += 1
            process_obj.ingest_frame(sempcd)

            runtime.publish_pose(latest_frame_index)
            runtime.publish_frame(latest_frame_index, sempcd)

            if not args.vector:
                continue
            if not process_obj.ready():
                continue

            logical_index = process_obj.logical_index(latest_frame_index)
            if logical_index < start_index:
                continue
            if logical_index >= args.max_index:
                print("reach max index, stop processing")
                break
            if not process_obj.should_process(logical_index):
                continue
            if process_obj.requires_pose and runtime.poses is None:
                print("trajectory is required for vectorization, skip processing")
                continue
            process_obj.process(runtime, logical_index)
            print(logical_index)
    except EOFError:
        print("done")
    except KeyboardInterrupt:
        print("interrupted by user")

    return np.zeros((0, 4), dtype=np.float32)
