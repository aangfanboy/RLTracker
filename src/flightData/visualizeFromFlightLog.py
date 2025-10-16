#!/usr/bin/env python3
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np

# Make project root importable when run directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    # Prefer tensorboard event accumulator if available
    from tensorboard.backend.event_processing import event_accumulator
    _HAS_EVENT_ACC = True
except Exception:
    event_accumulator = None
    _HAS_EVENT_ACC = False

from utils.math import floatMatrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def find_event_files(tb_dir: str) -> List[str]:
    if not os.path.isdir(tb_dir):
        return []
    files = []
    for fn in os.listdir(tb_dir):
        if fn.startswith("events.out"):
            files.append(os.path.join(tb_dir, fn))
    return sorted(files)


def load_scalars_from_event(tb_dir: str, tags: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """Return dict[tag] = list of (step, value) pairs sorted by step.
    Uses tensorboard.event_accumulator when available, else falls back to tensorflow.summary iterator.
    """
    result: Dict[str, List[Tuple[int, float]]] = {t: [] for t in tags}

    if _HAS_EVENT_ACC:
        ea = event_accumulator.EventAccumulator(tb_dir)
        try:
            ea.Reload()
        except Exception:
            # if reload fails, return empty
            return result
        
        allTags = ea.Tags().get('tensors', [])

        for tag in tags:
            try:
                items = ea.Tensors(tag)
                for item in items:
                    step = item.step
                    # value is stored as a tensor proto, extract float value
                    val = None
                    if item.tensor_proto and item.tensor_proto.tensor_content:
                        try:
                            val = float(np.frombuffer(item.tensor_proto.tensor_content, dtype=np.float32)[0])
                        except Exception:
                            val = None
                    if val is not None:
                        result[tag].append((step, val))
            except Exception:
                # tag not present
                pass

        # sort by step
        for tag in tags:
            result[tag].sort(key=lambda x: x[0])

        return result

    # Fallback: use tensorflow.summary iterator if tensorflow is installed
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.core.util.event_pb2 import Event
        from tensorflow.python.summary.summary_iterator import summary_iterator
    except Exception:
        return result

    files = find_event_files(tb_dir)
    for f in files:
        try:
            for e in summary_iterator(f):
                if not hasattr(e, 'summary') or e.summary is None:
                    continue
                step = int(getattr(e, 'step', 0))
                for v in e.summary.value:
                    tag = v.tag
                    if tag in result:
                        # value can be simple_number or tensor
                        val = None
                        if v.HasField('simple_value'):
                            val = float(v.simple_value)
                        else:
                            try:
                                # try to parse tensor proto fallback
                                val = float(np.frombuffer(v.tensor.tensor_content, dtype=np.float32)[0])
                            except Exception:
                                val = None
                        if val is not None:
                            result[tag].append((step, val))
        except Exception:
            continue

    for tag in tags:
        result[tag].sort(key=lambda x: x[0])

    return result


def assemble_flight_records(scalars: Dict[str, List[Tuple[int, float]]]) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Assemble records [(step, position(3,), orientation(4,))] by matching steps.
    FlightLogger stores position as flight/position/0..2 and orientation as flight/orientation/0..3
    """
    # Collect all steps present across tags
    steps = set()
    for v in scalars.values():
        for s, _ in v:
            steps.add(s)
    steps = sorted(steps)

    # Convert lists to dict for faster lookup
    tag_maps = {}
    for tag, arr in scalars.items():
        tag_maps[tag] = {s: val for s, val in arr}

    records = []
    for s in steps:
        try:
            timeStamp = tag_maps.get('flight/time', {}).get(s, np.nan)

            pos = np.array([
                tag_maps.get('flight/position/0', {}).get(s, np.nan),
                tag_maps.get('flight/position/1', {}).get(s, np.nan),
                tag_maps.get('flight/position/2', {}).get(s, np.nan),
            ], dtype=np.float64)

            orient = np.array([
                tag_maps.get('flight/orientation/0', {}).get(s, np.nan),
                tag_maps.get('flight/orientation/1', {}).get(s, np.nan),
                tag_maps.get('flight/orientation/2', {}).get(s, np.nan),
                tag_maps.get('flight/orientation/3', {}).get(s, np.nan),
            ], dtype=np.float64)

            target_pos = np.array([
                tag_maps.get('flight/target_position/0', {}).get(s, np.nan),
                tag_maps.get('flight/target_position/1', {}).get(s, np.nan),
                tag_maps.get('flight/target_position/2', {}).get(s, np.nan),
            ], dtype=np.float64)

            if np.isnan(pos).any() and np.isnan(orient).any():
                # skip incomplete record
                continue

            records.append((s, timeStamp, pos, orient, target_pos))
        except Exception:
            continue

    # Sort records by step
    records.sort(key=lambda x: x[0])
    return records


def run_replay(flight_dir: str, replay_rate_hz: float = 20.0):
    # Load map
    map_file = os.path.join(flight_dir, 'environment_map.npy')
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Map file not found: {map_file}")

    env_map = np.load(map_file)

    # Find tensorboard dir
    tb_dir = os.path.join(flight_dir, 'tensorboard')
    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")

    # Prepare tags we expect
    tags = ['flight/time'] + [f'flight/position/{i}' for i in range(3)] + [f'flight/orientation/{i}' for i in range(4)] + [f'flight/target_position/{i}' for i in range(3)]

    scalars = load_scalars_from_event(tb_dir, tags)
    records = assemble_flight_records(scalars)

    if len(records) == 0:
        print('No flight records found in tensorboard logs.')
        return

    records = [(s, pos, orient, target_pos) for (_, s, pos, orient, target_pos) in records]

    targetStaticPos = records[0][3]

    # draw the 3d map and a line showing the flight path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.imshow(env_map, extent=[-100, 100, -100, 100], alpha=0.5)
    path = np.array([pos for _, pos, _, _ in records])
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', label='Flight Path')
    ax.scatter([targetStaticPos[0]], [targetStaticPos[1]], [targetStaticPos[2]], color='red', marker='x', s=100, label='Target Position')
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='green', marker='o', s=100, label='Start Position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Flight Path Replay')
    ax.legend()
    plt.ion()
    plt.show(block=True)

def main():
    parser = argparse.ArgumentParser(description='Visualize flight log replay')
    parser.add_argument('flight_folder', help='Path to flight log folder (e.g. flightData/flight_... )')
    parser.add_argument('--hz', type=float, default=20.0, help='Replay rate in Hz')
    args = parser.parse_args()

    run_replay(args.flight_folder, replay_rate_hz=args.hz)


if __name__ == '__main__':
    main()
