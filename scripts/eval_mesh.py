import pathlib

import pymeshlab


def hausdorff_distance(source: pathlib.Path, target: pathlib.Path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(target.as_posix())
    ms.load_new_mesh(source.as_posix())
    res = ms.get_hausdorff_distance(targetmesh=0, sampledmesh=1)
    return res
