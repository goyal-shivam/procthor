import sys, os
# Ensure the LOCAL procthor package (not site-packages) is used for all imports
_procthor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _procthor_root not in sys.path:
    sys.path.insert(0, _procthor_root)

# Pin to a known-working Linux build commit (branch 'nanna'/'main' builds are no longer hosted)
# This must be set BEFORE the first procthor import.
from procthor.generation import PROCTHOR_INITIALIZATION
PROCTHOR_INITIALIZATION["commit_id"] = "ca10d107fb46cb051dba99af484181fda9947a28"
# Remove the branch key so the commit_id is used directly
PROCTHOR_INITIALIZATION.pop("branch", None)

# ## EXTRA CHANGES THAT I DID AFTER ASKING FROM CHATGPT ARE ABOVE THIS COMMENT

from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

house_generator = HouseGenerator(
    split="train", seed=42, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER
)
house, _ = house_generator.sample()
house.validate(house_generator.controller)

house.to_json("temp.json")
