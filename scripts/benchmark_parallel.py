"""
Benchmark parallel ProcTHOR house generation.

Tests N worker processes each with their own AI2-THOR controller (all on GPU 0),
and measures throughput to find the optimal parallelism for a single GPU.

Usage:
    conda run -n procthor-experiments python scripts/benchmark_parallel.py \
        --workers 1 2 4 --houses 5 --gpu-device 0
"""
import sys, os, time, argparse, json
from multiprocessing import Process, Queue

_procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _procthor_root not in sys.path:
    sys.path.insert(0, _procthor_root)


def worker_fn(worker_id: int, n_houses: int, result_queue: Queue):
    """Each worker initializes its own controller and generates n_houses.

    Uses the default platform (no CloudRendering) because:
      - Every controller.step() call uses renderImage=False
      - No GPU rendering is ever performed during house generation
      - CloudRendering adds ~65s extra init + per-step overhead
      - Default platform is ~1.7x faster overall
    """
    import os, sys, time, random
    _procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _procthor_root not in sys.path:
        sys.path.insert(0, _procthor_root)

    # Rebuild PROCTHOR_INITIALIZATION cleanly (forked processes may share state)
    from procthor import constants
    constants.PROCTHOR_INITIALIZATION.clear()
    constants.PROCTHOR_INITIALIZATION.update({
        "commit_id": "ca10d107fb46cb051dba99af484181fda9947a28",
        "scene": "Procedural",
    })
    # NOTE: No CloudRendering — default platform is faster for this workload

    from ai2thor.controller import Controller
    from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
    from procthor.constants import PROCTHOR_INITIALIZATION

    # --- init controller (one-time cost per worker) ---
    t0 = time.time()
    try:
        controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "error": str(e), "house_times": []})
        return
    t_init = time.time() - t0
    print(f"  [worker {worker_id}] controller_init: {t_init:.2f}s", flush=True)

    house_generator = HouseGenerator(
        split="train",
        seed=100 + worker_id * 1000,
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
        controller=controller,
    )

    # --- generate houses ---
    house_times = []
    for i in range(n_houses):
        t_house = time.time()
        # reseed each house so workers produce different houses
        house_generator.set_seed(random.randint(0, 2**31))
        house, _ = house_generator.sample()
        house.validate(house_generator.controller)
        elapsed = time.time() - t_house
        house_times.append(elapsed)
        print(f"  [worker {worker_id}] house {i+1}/{n_houses}: {elapsed:.2f}s", flush=True)

    try:
        controller.stop()
    except Exception:
        pass

    result_queue.put({
        "worker_id": worker_id,
        "init_time": t_init,
        "house_times": house_times,
        "total_generation_time": sum(house_times),
    })


def run_benchmark(n_workers: int, n_houses: int):
    print(f"\n{'='*60}")
    print(f"  Benchmark: {n_workers} workers x {n_houses} houses = {n_workers*n_houses} total")
    print(f"  Platform: default (no CloudRendering — faster for generation)")
    print(f"{'='*60}")

    q = Queue()
    t_wall_start = time.time()

    procs = [
        Process(target=worker_fn, args=(i, n_houses, q))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    t_wall_total = time.time() - t_wall_start

    results = []
    while not q.empty():
        results.append(q.get())

    errors = [r for r in results if "error" in r]
    if errors:
        for e in errors:
            print(f"  !! Worker {e['worker_id']} failed: {e['error']}")

    total_houses = sum(len(r["house_times"]) for r in results)
    all_house_times = [t for r in results for t in r["house_times"]]
    avg_per_house = sum(all_house_times) / len(all_house_times) if all_house_times else 0
    effective_rate = total_houses / t_wall_total if total_houses > 0 else 0
    effective_sec_per_house = t_wall_total / total_houses if total_houses > 0 else float('inf')

    print(f"\n  Results ({n_workers} workers):")
    print(f"    Total wall-clock time:      {t_wall_total:.2f}s")
    print(f"    Total houses generated:     {total_houses}")
    print(f"    Avg time per house (worker):{avg_per_house:.2f}s")
    print(f"    Effective houses/second:    {effective_rate:.4f}")
    print(f"    Effective seconds/house:    {effective_sec_per_house:.2f}s")
    print(f"{'='*60}")

    return {
        "n_workers": n_workers,
        "n_houses": total_houses,
        "wall_time": t_wall_total,
        "total_houses": total_houses,
        "effective_rate": effective_rate,
        "effective_sec_per_house": effective_sec_per_house,
        "avg_per_house_per_worker": avg_per_house,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2],
                        help="Number of workers to benchmark (list)")
    parser.add_argument("--houses", type=int, default=3,
                        help="Houses to generate per worker")
    args = parser.parse_args()

    all_results = []
    for n_workers in args.workers:
        r = run_benchmark(n_workers, args.houses)
        all_results.append(r)

    # Print summary comparison
    if len(all_results) > 1:
        baseline = all_results[0]
        print(f"\n{'='*60}")
        print(f"  SPEEDUP SUMMARY (baseline: {baseline['n_workers']} worker(s))")
        print(f"{'='*60}")
        print(f"  {'Workers':>8} | {'sec/house':>12} | {'houses/sec':>12} | {'Speedup':>10}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for r in all_results:
            speedup = baseline["effective_sec_per_house"] / r["effective_sec_per_house"]
            print(f"  {r['n_workers']:>8} | {r['effective_sec_per_house']:>12.2f} | {r['effective_rate']:>12.4f} | {speedup:>10.2f}x")
        print(f"{'='*60}")

    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to /tmp/benchmark_results.json")
