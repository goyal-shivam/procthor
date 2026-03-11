"""
Multi-GPU ProcTHOR house generation using SSH + multiprocessing.

Strategy:
  - Each server runs its own set of worker processes
  - Workers on Server A use GPU(s) on Server A
  - Workers on Server B use GPU(s) on Server B
  - Results are saved to a shared directory OR transferred back

This script is designed to run on EACH server independently.
Use a simple coordinator (run_all_servers.sh) to launch it on both.

Usage (run this on each server separately):
    # On Server A (1 GPU):
    conda run -n procthor-experiments python scripts/generate_multi_server.py \
        --server-id A --workers-per-gpu 4 --gpu-devices 0 \
        --total-houses 1000 --output-dir /shared/houses

    # On Server B (2 GPUs):
    conda run -n procthor-experiments python scripts/generate_multi_server.py \
        --server-id B --workers-per-gpu 4 --gpu-devices 0 1 \
        --total-houses 2000 --output-dir /shared/houses

Architecture:
    - workers-per-gpu workers are launched per GPU
    - Each worker gets its AI2-THOR controller pinned to its assigned GPU
      via CloudRendering + gpu_device (needed ONLY for multi-GPU assignment)
    - Controller init cost is paid once per worker, amortized across houses
"""
import sys, os, time, argparse, random, json
from multiprocessing import Pool, Value, Lock
from pathlib import Path

_procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _procthor_root not in sys.path:
    sys.path.insert(0, _procthor_root)

_counter = None
_counter_lock = None
_gpu_device = None
_output_dir = None
_server_id = None
_split = None


def init_worker(counter, lock, gpu_device, output_dir, server_id, split):
    global _counter, _counter_lock, _gpu_device, _output_dir, _server_id, _split
    _counter = counter
    _counter_lock = lock
    _gpu_device = gpu_device
    _output_dir = output_dir
    _server_id = server_id
    _split = split


_house_generator = None


def _get_house_generator():
    global _house_generator
    if _house_generator is not None:
        return _house_generator

    from procthor.generation import PROCTHOR_INITIALIZATION
    PROCTHOR_INITIALIZATION["commit_id"] = "ca10d107fb46cb051dba99af484181fda9947a28"
    PROCTHOR_INITIALIZATION.pop("branch", None)

    pid = os.getpid()
    use_cloud = _gpu_device is not None

    if use_cloud:
        # CloudRendering is needed ONLY to pin to a specific GPU device.
        # If you only have 1 GPU and don't care which it uses, default is fine.
        from ai2thor.platform import CloudRendering
        PROCTHOR_INITIALIZATION["platform"] = CloudRendering
        PROCTHOR_INITIALIZATION["gpu_device"] = _gpu_device
        print(f"  [Server {_server_id} PID {pid}] Init on GPU {_gpu_device}...", flush=True)
    else:
        print(f"  [Server {_server_id} PID {pid}] Init (default platform)...", flush=True)

    from ai2thor.controller import Controller
    from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

    t0 = time.time()
    controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)
    print(f"  [Server {_server_id} PID {pid}] Controller ready in {time.time() - t0:.1f}s", flush=True)

    _house_generator = HouseGenerator(
        split=_split,
        seed=pid,
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
        controller=controller,
    )
    return _house_generator


def generate_one_house(task_i: int):
    hg = _get_house_generator()
    seed = random.randint(0, 2**31)
    hg.set_seed(seed)

    t0 = time.time()
    while True:
        house, _ = hg.sample()
        house.validate(hg.controller)
        if not house.data["metadata"]["warnings"]:
            break
        hg.room_spec = house.room_spec

    elapsed = time.time() - t0

    with _counter_lock:
        _counter.value += 1
        idx = _counter.value

    fname = f"server{_server_id}_gpu{_gpu_device}_house{idx:06d}_seed{seed}.json"
    out_path = os.path.join(_output_dir, fname)
    house.to_json(out_path)
    print(f"  [{os.getpid()}] GPU{_gpu_device} House {idx} done in {elapsed:.2f}s", flush=True)
    return task_i, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-id", type=str, required=True,
                        help="Unique label for this server (e.g., A or B)")
    parser.add_argument("--workers-per-gpu", type=int, default=4,
                        help="Worker processes per GPU")
    parser.add_argument("--gpu-devices", type=int, nargs="+", default=[0],
                        help="GPU device indices to use on this server")
    parser.add_argument("--total-houses", type=int, default=100,
                        help="Total houses to generate on this server")
    parser.add_argument("--output-dir", type=str, default="outputs/houses")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    total_workers = args.workers_per_gpu * len(args.gpu_devices)

    print(f"Server {args.server_id}: {total_workers} workers across GPUs {args.gpu_devices}")
    print(f"Generating {args.total_houses} houses → {args.output_dir}")
    print()

    # Build per-GPU pools
    # Each GPU gets its own Pool, tasks are distributed evenly
    houses_per_gpu = args.total_houses // len(args.gpu_devices)
    all_results = []
    t_start = time.time()

    # We launch all GPU pools approximately simultaneously using processes
    from multiprocessing import Process, Queue

    def run_gpu_pool(gpu_device: int, n_houses: int, result_q: Queue):
        counter = Value("i", 0)
        lock = Lock()
        with Pool(
            processes=args.workers_per_gpu,
            initializer=init_worker,
            initargs=(counter, lock, gpu_device, args.output_dir, args.server_id, args.split),
        ) as pool:
            results = pool.map(generate_one_house, range(n_houses))
        result_q.put(results)

    rq = Queue()
    gpu_procs = []
    for i, gpu_id in enumerate(args.gpu_devices):
        n = houses_per_gpu if i < len(args.gpu_devices) - 1 else args.total_houses - houses_per_gpu * i
        p = Process(target=run_gpu_pool, args=(gpu_id, n, rq))
        p.start()
        gpu_procs.append(p)

    for p in gpu_procs:
        p.join()

    while not rq.empty():
        all_results.extend(rq.get())

    total_wall = time.time() - t_start
    gen_times = [r[1] for r in all_results]

    print(f"\n{'='*60}")
    print(f"  Server {args.server_id} DONE")
    print(f"  Total houses:         {args.total_houses}")
    print(f"  Wall-clock time:      {total_wall:.1f}s")
    print(f"  Workers:              {total_workers} ({args.workers_per_gpu}/GPU × {len(args.gpu_devices)} GPUs)")
    print(f"  Avg per house:        {sum(gen_times)/len(gen_times):.2f}s (worker time)")
    print(f"  Effective rate:       {args.total_houses / total_wall:.3f} houses/sec")
    print(f"  Effective sec/house:  {total_wall / args.total_houses:.2f}s")
    print(f"{'='*60}")

    with open(f"/tmp/results_server{args.server_id}.json", "w") as f:
        json.dump({"wall_time": total_wall, "total_houses": args.total_houses,
                   "gen_times": gen_times}, f, indent=2)


if __name__ == "__main__":
    main()
