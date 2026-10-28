"""Deterministic seeds for CrowdCC corruptions."""

from __future__ import annotations

from hashlib import blake2b

import numpy as np

SEED_NAMESPACE = "crowd-counting-c/seeds/v1"

DEFAULT_MASTER_SEED = 1234

_SEED_BYTES = 8


def _digest_seed(*parts: object) -> int:
    """Derive a 64-bit seed from U+001F-delimited parts."""
    payload = "\x1f".join([SEED_NAMESPACE, *(str(p) for p in parts)])
    digest = blake2b(payload.encode("utf-8"), digest_size=_SEED_BYTES).digest()
    return int.from_bytes(digest, "big")


def sensor_seed(
    dataset: str,
    corruption: str,
    level: str,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> int:
    """Seed fixed sensor defects shared across a dataset."""
    return _digest_seed("sensor", master_seed, dataset, corruption, level)


def image_seed(
    image_id: str,
    corruption: str,
    level: str,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> int:
    """Seed per-image randomness without hashing the dataset."""
    return _digest_seed("image", master_seed, image_id, corruption, level)


def make_rngs(
    dataset: str,
    image_id: str,
    corruption: str,
    level: str,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> tuple[np.random.Generator, np.random.Generator]:
    """Return sensor and image RNGs for a corruption cell."""
    rng_sensor = np.random.default_rng(sensor_seed(dataset, corruption, level, master_seed))
    rng_image = np.random.default_rng(image_seed(image_id, corruption, level, master_seed))
    return rng_sensor, rng_image


def seeding_policy(master_seed: int = DEFAULT_MASTER_SEED) -> dict[str, object]:
    """Describe the seeding scheme for cache manifests."""
    return {
        "namespace": SEED_NAMESPACE,
        "master_seed": master_seed,
        "hash": f"blake2b(digest_size={_SEED_BYTES}) over "
                f"'{SEED_NAMESPACE}' + parts joined by U+001F",
        "sensor_seed": "sensor_seed(dataset, corruption, level, master_seed): "
                       "constant across images; fixed hardware defects only",
        "image_seed": "image_seed(image_id, corruption, level, master_seed): "
                      "per image; all shot-to-shot randomness",
        "realizations": "one fixed realization for the full sweep; change "
                        "master_seed to generate another realization",
    }
