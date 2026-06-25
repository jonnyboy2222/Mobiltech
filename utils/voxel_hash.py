import itertools
import numpy as np


def get_neighbor_keys(key, radius=1):
    ix, iy, iz = key

    offsets = itertools.product(
        range(-radius, radius + 1),
        range(-radius, radius + 1),
        range(-radius, radius + 1),
    )

    return [(ix + dx, iy + dy, iz + dz) for dx, dy, dz in offsets]


def query_neighbor_voxels(voxel_map, key, radius=1):
    neighbors = []

    for nk in get_neighbor_keys(key, radius):
        if nk in voxel_map:
            neighbors.append(voxel_map[nk])

    return neighbors