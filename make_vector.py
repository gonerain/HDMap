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
    parser.add_argument("--max_index", default=10000, type=int, help="Max logical index to process")
    parser.add_argument("--start_index", default=None, type=int, help="Start processing from this logical frame index")
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
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted by user")
