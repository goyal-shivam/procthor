import sys, os, time
start_time = time.time()

# Ensure the LOCAL procthor package (not site-packages) is used for all imports
_procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _procthor_root not in sys.path:
    sys.path.insert(0, _procthor_root)

from ai2thor.platform import CloudRendering
from procthor.utils.types import SamplingVars

# Pin to a known-working Linux build commit (branch 'nanna'/'main' builds are no longer hosted)
# This must be set BEFORE the first procthor import.
from procthor.generation import PROCTHOR_INITIALIZATION
PROCTHOR_INITIALIZATION["commit_id"] = "ca10d107fb46cb051dba99af484181fda9947a28"
# Remove the branch key so the commit_id is used directly
PROCTHOR_INITIALIZATION.pop("branch", None)
# Use the NVIDIA GPU via EGL headless rendering (no Xvfb needed)
# PROCTHOR_INITIALIZATION["platform"] = CloudRendering

# ## EXTRA CHANGES THAT I DID AFTER ASKING FROM CHATGPT ARE ABOVE THIS COMMENT

from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

house_generator = HouseGenerator(
    split="train", seed=310326, room_spec="5-room"
)


house, _ = house_generator.sample(
    sampling_vars=SamplingVars(
        interior_boundary_scale=2.0,  # range is 1.6–2.2 normally
        max_floor_objects=30,
    )
)
house.validate(house_generator.controller)

# house.to_json("temp.json")
house.to_json("31-march-2026_2.json")

print(f"Total execution time: {time.time() - start_time:.4f} seconds")
