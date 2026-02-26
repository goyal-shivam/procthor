from procthor.generation import PROCTHOR_INITIALIZATION
# PROCTHOR_INITIALIZATION["commit_id"] = "455cf72a1c8e0759a452422f2128fbc93a3cb06b"
# # (optional) stay on the default 'main' branch
# PROCTHOR_INITIALIZATION["branch"] = "main"

# Pin to a known-working Linux build commit (branch 'nanna'/'main' builds are no longer hosted)
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
