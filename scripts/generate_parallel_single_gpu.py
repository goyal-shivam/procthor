"""
Parallel ProcTHOR house generation — Single GPU, multiple workers.

Strategy: Use multiprocessing.Pool with N workers. Each worker:
  1. Creates its own AI2-THOR Controller (default platform, no CloudRendering
     since renderImage=False everywhere — no GPU needed for generation)
  2. Reuses that controller for many houses (amortizes the 72s init cost)

Usage:
    conda run -n procthor-experiments python scripts/generate_parallel_single_gpu.py \
        --workers 4 --total-houses 100 --output-dir outputs/houses

Why NOT CloudRendering?
    - All controller steps use renderImage=False → GPU is never used
    - CloudRendering adds ~65s extra init + overhead per step
    - Default platform is faster for pure structure generation
"""
import sys, os, time, argparse, random
from multiprocessing import Pool, Value, Lock
from pathlib import Path

_procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _procthor_root not in sys.path:
    sys.path.insert(0, _procthor_root)

# ── Shared counter across workers ──
_counter = None
_counter_lock = None
_output_dir = None
_split = None


def init_worker(counter, counter_lock, output_dir, split):
    """Called once when each worker process starts."""
    global _counter, _counter_lock, _output_dir, _split
    _counter = counter
    _counter_lock = counter_lock
    _output_dir = output_dir
    _split = split


# Per-worker state (lazy init, reused across tasks)
_house_generator = None


def _get_house_generator():
    global _house_generator
    if _house_generator is not None:
        return _house_generator

    from procthor.generation import PROCTHOR_INITIALIZATION
    PROCTHOR_INITIALIZATION["commit_id"] = "ca10d107fb46cb051dba99af484181fda9947a28"
    PROCTHOR_INITIALIZATION.pop("branch", None)
    # DO NOT set CloudRendering — it's slower for this workload
    # (generation never renders images)

    from ai2thor.controller import Controller
    from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

    pid = os.getpid()
    print(f"  [PID {pid}] Initializing controller...", flush=True)
    t0 = time.time()
    controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)
    print(f"  [PID {pid}] Controller ready in {time.time() - t0:.1f}s", flush=True)

    _house_generator = HouseGenerator(
        split=_split,
        seed=pid,
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
        controller=controller,
    )
    return _house_generator


def generate_one_house(task_i: int):
    """Generate one house and save it. Returns (task_i, elapsed, warnings)."""
    hg = _get_house_generator()
    seed = random.randint(0, 2**31)
    hg.set_seed(seed)

    t0 = time.time()
    while True:
        house, _ = hg.sample()
        house.validate(hg.controller)
        if not house.data["metadata"]["warnings"]:
            break
        # keep same room_spec for unbiased sampling on retry
        hg.room_spec = house.room_spec

    elapsed = time.time() - t0

    with _counter_lock:
        _counter.value += 1
        idx = _counter.value

    out_path = os.path.join(_output_dir, f"house_{idx:06d}_seed{seed}.json")
    house.to_json(out_path)
    print(f"  [{os.getpid()}] House {idx} done in {elapsed:.2f}s → {out_path}", flush=True)
    return task_i, elapsed, house.data["metadata"]["warnings"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--total-houses", type=int, default=20,
                        help="Total number of houses to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/houses")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.total_houses} houses with {args.workers} workers")
    print(f"Output: {args.output_dir}")
    print(f"Note: CloudRendering is DISABLED (faster for pure generation)")
    print()

    counter = Value("i", 0)
    lock = Lock()

    t_start = time.time()
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(counter, lock, args.output_dir, args.split),
    ) as pool:
        results = pool.map(generate_one_house, range(args.total_houses))

    total_wall = time.time() - t_start
    gen_times = [r[1] for r in results]

    print(f"\n{'='*55}")
    print(f"  DONE — {args.total_houses} houses in {total_wall:.1f}s wall-clock")
    print(f"  Workers:              {args.workers}")
    print(f"  Avg per house:        {sum(gen_times)/len(gen_times):.2f}s (worker time)")
    print(f"  Effective rate:       {args.total_houses / total_wall:.3f} houses/sec")
    print(f"  Effective sec/house:  {total_wall / args.total_houses:.2f}s")
    print(f"  Output:               {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
