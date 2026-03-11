"""Profile ProcTHOR house generation to identify time bottlenecks.

Run with:
    python scripts/profile_generation.py          # without CloudRendering
    python scripts/profile_generation.py --cloud   # with CloudRendering
"""
import sys, os, time, argparse

# Ensure the LOCAL procthor package is used
_procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _procthor_root not in sys.path:
    sys.path.insert(0, _procthor_root)

parser = argparse.ArgumentParser()
parser.add_argument("--cloud", action="store_true", help="Enable CloudRendering")
parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index")
args = parser.parse_args()

# --- Timing helper ---
timings = {}
def timed(label):
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *a):
            elapsed = time.time() - self.t0
            timings[label] = elapsed
            print(f"  [{label}] {elapsed:.3f}s")
    return Timer()

# === Phase 1: Imports ===
with timed("import_procthor"):
    from procthor.generation import PROCTHOR_INITIALIZATION
    PROCTHOR_INITIALIZATION["commit_id"] = "ca10d107fb46cb051dba99af484181fda9947a28"
    PROCTHOR_INITIALIZATION.pop("branch", None)

    if args.cloud:
        from ai2thor.platform import CloudRendering
        PROCTHOR_INITIALIZATION["platform"] = CloudRendering
        PROCTHOR_INITIALIZATION["gpu_device"] = args.gpu_device
        print("CloudRendering ENABLED, gpu_device =", args.gpu_device)
    else:
        print("CloudRendering DISABLED (default platform)")

    from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

# === Phase 2: Controller init ===
from ai2thor.controller import Controller
with timed("controller_init"):
    controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)

# === Phase 3: House generation ===
with timed("house_generator_create"):
    house_generator = HouseGenerator(
        split="train", seed=182,
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
        controller=controller,
    )

with timed("house_sample"):
    house, _ = house_generator.sample()

# === Phase 4: Validation ===
with timed("house_validate"):
    house.validate(house_generator.controller)

# === Phase 5: Save ===
with timed("house_to_json"):
    house.to_json("temp.json")

# === Summary ===
total = sum(timings.values())
print(f"\n{'='*50}")
print(f"TIMING SUMMARY ({'CloudRendering' if args.cloud else 'Default platform'}):")
print(f"{'='*50}")
for label, elapsed in timings.items():
    pct = (elapsed / total) * 100
    print(f"  {label:30s} {elapsed:8.3f}s  ({pct:5.1f}%)")
print(f"  {'TOTAL':30s} {total:8.3f}s")
print(f"{'='*50}")
