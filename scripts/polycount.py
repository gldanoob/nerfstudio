import os

import trimesh

datasets = [
    'money_no_vase',
    'tree_no_vase',
    'palm',
    'palm2',
    'bush',
    'agapanthus',
    'grass'
]

dataset_dir = 'exports/final'

mesh_filename = 'poisson_mesh.ply'

for dataset in datasets:
    mesh_path = os.path.join(dataset_dir, dataset, mesh_filename)
    if os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path)
        print(f"Dataset: {dataset}, Poly Count: {len(mesh.faces)}")
    else:
        print(f"Mesh file not found for dataset: {dataset}")