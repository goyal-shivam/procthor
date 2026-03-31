# Increasing House Variety in ProcTHOR

## 1. Change the `seed` (most impactful)

Every seed produces a completely different house. The seed controls:
- Which room spec layout gets sampled (via weighted random choice)
- Interior boundary scale (drawn from `uniform(1.6, 2.2)`)
- All object/door/light placements

```python
# Change seed=182 to any other integer, or omit it entirely for random
house_generator = HouseGenerator(
    split="train", seed=42, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER
)
```

To **batch generate** multiple houses, loop over seeds:

```python
for seed in range(100):
    house_generator = HouseGenerator(split="train", seed=seed, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER)
    house, _ = house_generator.sample()
    house.to_json(f"houses/house_{seed}.json")
```

---

## 2. Pin a specific room layout with `room_spec`

Instead of letting the sampler randomly pick, you can force a specific layout. The 16 available layouts (from `PROCTHOR10K_ROOM_SPEC_SAMPLER`) are:

| Layout | Description |
|---|---|
| `"4-room"` | Most common (weight 5x) — living + kitchen + 2 rooms |
| `"bedroom-bathroom"` | Small 2-room |
| `"kitchen-living-room"` | Open plan |
| `"kitchen"` / `"living-room"` / `"bedroom"` / `"bathroom"` | Single room |
| `"2-bed-1-bath"` / `"2-bed-2-bath"` | Apartment style |
| `"5-room"` / `"7-room-3-bed"` / `"8-room-3-bed"` / `"12-room"` / `"12-room-3-bed"` | Larger houses |

```python
# Force a specific layout (ignores the sampler)
house_generator = HouseGenerator(
    split="train", seed=42, room_spec="12-room-3-bed"
)
```

---

## 3. Use a different `split`

Splits affect which assets (objects, textures) are available:

```python
house_generator = HouseGenerator(
    split="val",   # or "test", "train"
    seed=42, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER
)
```

---

## 4. Use `SamplingVars` for size/density control

The `sample()` method accepts `sampling_vars` to override defaults:

```python
from procthor.generation import SamplingVars  # check exact import path

house, _ = house_generator.sample(
    sampling_vars=SamplingVars(
        interior_boundary_scale=2.0,  # range is 1.6–2.2 normally
        max_floor_objects=30,         # controls object density
    )
)
```

---

## Quick recipe for maximum variety

The two highest-impact changes in `example.py`:

1. **Vary the seed** — different seed = completely different house
2. **Vary the room_spec or let the sampler pick** — different layout template

> **Note:** The `"4-room"` spec is sampled 5x more often than others by default,
> so if you want variety in *layout*, either pin a specific spec per run or accept
> that ~30% of random generations will be 4-room houses.

---

## HouseGenerator parameters reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `split` | `"train"/"val"/"test"` | required | Asset split to use |
| `seed` | `int` or `None` | `None` (random) | RNG seed for reproducibility |
| `room_spec` | `str` or `RoomSpec` | `None` | Pin a specific layout |
| `room_spec_sampler` | `RoomSpecSampler` | `None` | Sampler to pick layout randomly |
| `interior_boundary` | `np.array` | `None` | Override house footprint |
| `controller` | `Controller` | `None` | Reuse an existing AI2-THOR controller |
| `pt_db` | `ProcTHORDatabase` | default DB | Asset database |

## sample() parameters reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `partial_house` | `PartialHouse` | `None` | Resume from a partial house |
| `return_partial_houses` | `bool` | `False` | Return intermediate states per stage |
| `sampling_vars` | `SamplingVars` | `None` | Override scale/density defaults |
| `next_sampling_stage` | `NextSamplingStage` | `STRUCTURE` | Starting stage for sampling |
