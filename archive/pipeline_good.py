import copy
import math
import pathlib
import random
from typing import Tuple

import eval_mesh
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pymeshlab.pmeshlab as pmeshlab
import trimesh
from matplotlib import cm
from sklearn.cluster import DBSCAN, spectral_clustering
from sklearn.kernel_approximation import pairwise_kernels
from sklearn.neighbors import KNeighborsClassifier

CACHING = False
VERBOSE = True
EXPORT_CLUSTERS = True
RANDOM_STATE = 42

# Hyperparameters
T_SIGMA = 7
N_TRIANGLES = 150

N_CLEAN = 500_000
N_CORE = 60
EPS = 0.08

N_POINTS = 10000
N_CLUSTERS = 170
GAMMA = 30

N_LEAF_TRIANGLES = 100
T_VAR = 0.55
EVAL_LEAF_HEURISTIC = True

N_SMOOTH = 20
ALPHA_SMOOTH = 0.0

N_HOLES = 200

ALPHA_WRAP = 1/200
OFFSET_WRAP = 1/100000

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def lprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def custom_draw_geometry(mesh, **kwargs):
    mesh_ = copy.deepcopy(mesh)
    mesh_.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_], **kwargs)

def plot_vertices(vertices):
    assert vertices.shape[1] == 3
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    plt.show()

def generate_colormap(N):
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up)
    arr = arr.reshape(7, N_up//7).T.reshape(-1)
    ret = cm.hsv(arr)
    n = ret[:, 3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2, i] *= np.arange(0.2, 1, 0.8/a)
    ret[n//2:, 3] *= np.arange(1, 0.1, -0.9/b)
    return ret

def pymesh_to_trimesh(mesh: pmeshlab.Mesh):
    v = np.asarray(mesh.vertex_matrix())
    f = np.asarray(mesh.face_matrix())
    m = trimesh.Trimesh(v, f, process=False)
    assert (m.faces == f).all(), 'Face matrix does not match'
    return m

def trimesh_to_pymesh(mesh: trimesh.Trimesh):
    v = mesh.vertices
    f = mesh.faces
    m = pmeshlab.Mesh(vertex_matrix=v, face_matrix=f)
    assert (m.face_matrix() == f).all(), 'Face matrix does not match'
    return m

class Pipeline():
    def __init__(self, in_path: pathlib.Path, out_path: pathlib.Path, gt_dir: pathlib.Path):
        self.in_path = in_path
        self.out_path = out_path
        self.gt_dir = gt_dir
        self.tmp_path = 'tmp/tmp.ply'

    def run(self, cleaning: str, clustering: str, smoothing: str, post_processing: str, stop_at: str = 'all'):
        m: trimesh.Trimesh = trimesh.load_mesh(self.in_path.as_posix(), process=False)

        if CACHING:
            m = trimesh.load_mesh(self.out_path.parent / f'{cleaning}.ply', process=False)
            lprint(f'Loaded {cleaning}.ply from cache')
        elif cleaning == 'z-score':
            lprint('Removing clusters with high z-score')
            m = self.remove_clusters_with_high_z_score(m, threshold=T_SIGMA, n_triangles=N_TRIANGLES)
        
        elif cleaning == 'dbscan':
            lprint('Removing clusters with dbscan')
            m = self.remove_with_dbscan(m, eps=EPS, min_samples=N_CORE, n_samples=N_CLEAN)

        elif cleaning == 'none':
            lprint('No cleaning (process only)')
            m = trimesh.Trimesh(m.vertices, m.faces, process=True)
            pass

        else:
            raise ValueError('Unknown cleaning method')

        ms = pmeshlab.MeshSet(verbose=VERBOSE)
        ms.add_mesh(trimesh_to_pymesh(m))

        m_cleaned: pmeshlab.Mesh = ms.current_mesh()

        if stop_at == 'cleaning':
            ms.save_current_mesh(self.out_path.as_posix())
            return

        if stop_at != 'smoothing' and CACHING and (self.out_path.parent / f'{smoothing}.ply').exists():
            ms_new = pmeshlab.MeshSet(verbose=VERBOSE)
            lprint(f'Loaded {smoothing}.ply from cache')
            ms_new.load_new_mesh((self.out_path.parent / f'{smoothing}.ply').as_posix())

        else:
            # Clustering
            if clustering == 'sample':
                _, clustering_all = self.cluster_sample(m_cleaned, N_CLUSTERS, gamma=30, n_samples=N_POINTS)

            elif clustering == 'decimate':
                _, clustering_all = self.cluster_decimate(m_cleaned, N_CLUSTERS, n_samples=N_POINTS)

            else: 
                raise ValueError('Unknown clustering method')
                
            if stop_at == 'clustering':
                self.plot_clusterings(m_cleaned.vertex_matrix(), clustering_all, N_CLUSTERS)
                return

            # get submesh for each cluster
            main_trimesh = pymesh_to_trimesh(m_cleaned)

            # process each cluster
            ms_new = pmeshlab.MeshSet(verbose=VERBOSE)
            ms_tmp = pmeshlab.MeshSet(verbose=VERBOSE)
            ms_eval = pmeshlab.MeshSet(verbose=VERBOSE)
            clustered_faces = np.zeros(main_trimesh.faces.shape[0], dtype=bool)
            for i in range(N_CLUSTERS):
                lprint('Getting cluster #', i)
                no_vertices = np.sum(clustering_all == i)
                lprint('Number of vertices:', no_vertices)
                if no_vertices < 3:
                    lprint('Skipping cluster with less than 3 vertices')
                    continue

                cluster_indices = np.where(clustering_all == i)[0]
                subfaces = np.any(np.isin(main_trimesh.faces, cluster_indices), axis=1)
                # Remove faces that are already in previous clusters
                subfaces = np.logical_and(subfaces, ~clustered_faces)
                # mark faces as clustered
                clustered_faces[subfaces] = True

                cluster_mesh = main_trimesh.submesh([subfaces], append=True)
                if not isinstance(cluster_mesh, trimesh.Trimesh):
                    lprint('Cluster mesh is not a trimesh')
                    continue

                # split cluster into connected components
                labels = trimesh.graph.connected_component_labels(cluster_mesh.face_adjacency, cluster_mesh.faces.shape[0])
                assert labels.shape[0] == cluster_mesh.faces.shape[0], \
                'Labels shape does not match faces shape: ' + str(labels.shape) + ' != ' + str(cluster_mesh.faces.shape)
                lprint('Number of components:', labels.max() + 1)

                subfaces_list = []
                for j in range(labels.max() + 1):
                    subfaces = labels == j
                    subfaces_list.append(subfaces)

                subfaces_list.sort(key=lambda x: x.sum(), reverse=True)
                cluster_meshes: list[trimesh.Trimesh] = cluster_mesh.submesh(subfaces_list, append=False)

                for j, subcluster_mesh in enumerate(cluster_meshes):
                    if j < 10 and self.is_leaf_heuristic(subcluster_mesh):
                        lprint('Smoothing leaf cluster #', i)
                        if stop_at == 'heuristic':
                            ms_eval.add_mesh(trimesh_to_pymesh(subcluster_mesh))
                        else:
                            smoothed = self.smooth_leaf(ms_tmp, subcluster_mesh, N_SMOOTH, ALPHA_SMOOTH)
                            ms_new.add_mesh(smoothed)
                        if EXPORT_CLUSTERS:
                            subcluster_mesh.export(f'tmp/cluster_{i}_{j}.ply')
                    else:
                        ms_new.add_mesh(trimesh_to_pymesh(subcluster_mesh))


            if stop_at == 'heuristic':
                ms_eval.generate_by_merging_visible_meshes()
                return

            # combine all submeshes
            ms_new.generate_by_merging_visible_meshes()
            ms_new.save_current_mesh((self.out_path.parent / 'no_cleaning.ply').as_posix())
            ms_new.meshing_repair_non_manifold_vertices()
            ms_new.meshing_repair_non_manifold_edges()

            if stop_at == 'smoothing':
                ms_new.save_current_mesh(self.out_path.as_posix())
                return


        if post_processing == 'alpha_wrap':
            ms_new.generate_alpha_wrap(alpha_fraction=ALPHA_WRAP, offset_fraction=OFFSET_WRAP)

        elif post_processing == 'close_holes':
            # run close holes method
            ms_new.meshing_close_holes(maxholesize=N_HOLES, newfaceselected=False, refinehole=True)

        if stop_at == 'all':
            ms_new.save_current_mesh(self.out_path.as_posix())

    def remove_clusters_with_high_z_score(self, mesh: trimesh.Trimesh, threshold: float, n_triangles) -> trimesh.Trimesh:
        # Calculate distance from mean
        vertices = mesh.vertices
        faces = mesh.faces
        mean = np.mean(vertices, axis=0)
        std = np.std(np.linalg.norm(vertices - mean, axis=1), axis=0)

        print('Mean:', mean)
        print('Std:', std)

        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        unique_labels, counts = np.unique(labels, return_counts=True)

        labels_to_remove = []
        for label in unique_labels:
            cluster_faces = faces[labels == label]
            cluster_vertices = vertices[cluster_faces.flatten()]
            mean_position = np.mean(cluster_vertices, axis=0)

            distance = np.linalg.norm(mean_position - mean)
            z_score = distance / std
            if z_score > threshold and counts[label] < n_triangles:
                labels_to_remove.append(label)


        triangles_to_remove = np.isin(labels, labels_to_remove)
        triangles_to_keep = ~triangles_to_remove

        mesh_0 = mesh.submesh([triangles_to_keep], append=True)

        assert isinstance(mesh_0, trimesh.Trimesh), \
            'Mesh is not a trimesh'

        return mesh_0

    def remove_with_dbscan(self, mesh: trimesh.Trimesh, n_samples, eps, min_samples) -> trimesh.Trimesh:
        points = mesh.vertices

        # Sample 75% of points
        sample = np.random.choice(points.shape[0], n_samples, replace=False)
        sample = points[sample]
    
        lprint('Sampled', sample.shape[0], 'vertices for cleaning')

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(sample)

        # Get the labels for the rest of the points
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(sample, labels)
        labels = classifier.predict(points)


        outlier_indices = np.where(labels == -1)[0]
        print('Number of outliers:', len(outlier_indices))
        outlier_subfaces = np.any(np.isin(mesh.faces, outlier_indices), axis=1)

        print(outlier_subfaces.sum(), 'faces removed')

        mesh_0 = mesh.submesh([~outlier_subfaces], append=True)
        assert isinstance(mesh_0, trimesh.Trimesh), \
            'Mesh is not a trimesh'

        return mesh_0

    def cluster_sample(self, m: pmeshlab.Mesh, n_clusters: int, gamma: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        # sample vertice
        X = np.asarray(m.vertex_matrix())
        sample = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[sample]
        lprint('Sampled', X.shape[0], 'vertices for clustering')

        # spectral clustering on sample
        affinity = pairwise_kernels(X, metric='rbf', gamma=gamma)
        clustering = spectral_clustering(
            affinity=affinity, n_clusters=n_clusters, assign_labels="cluster_qr", random_state=RANDOM_STATE,

        )

        # KNN on the rest of the vertices
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(X, clustering)

        X_all = np.asarray(m.vertex_matrix())
        lprint('Clustering the rest of the vertices')
        clustering_all = classifier.predict(X_all)

        return X_all, clustering_all

    def cluster_decimate(self, m: pmeshlab.Mesh, n_clusters: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        # decimate mesh
        lprint('Decimating mesh')
        X_all = np.asarray(m.vertex_matrix())
        target_face_num = int(n_samples / m.vertex_number() * m.face_number())

        ms = pmeshlab.MeshSet(verbose=VERBOSE)
        ms.add_mesh(m)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_face_num)


        m_new = ms.current_mesh()
        faces = np.asarray(m_new.face_matrix())
        no_vertices = m_new.vertex_number()

        # adjacency matrix
        A = np.zeros((no_vertices, no_vertices))
        for i in range(faces.shape[0]):
            A[faces[i, 0], faces[i, 1]] = 1
            A[faces[i, 1], faces[i, 0]] = 1
            A[faces[i, 1], faces[i, 2]] = 1
            A[faces[i, 2], faces[i, 1]] = 1
            A[faces[i, 2], faces[i, 0]] = 1
            A[faces[i, 0], faces[i, 2]] = 1

        # spectral clustering on all vertices
        clustering = spectral_clustering(
            affinity=A, n_clusters=n_clusters, assign_labels="cluster_qr", random_state=RANDOM_STATE
        )

        X = np.asarray(m_new.vertex_matrix())

        # KNN on the rest of the vertices
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(X, clustering)

        lprint('Clustering the rest of the vertices')
        clustering_all = classifier.predict(X_all)

        return X_all, clustering_all

    def is_leaf_heuristic(self, mesh: trimesh.Trimesh) -> bool:
        if mesh.faces.shape[0] < N_LEAF_TRIANGLES:
            return False

        face_normals = mesh.face_normals.copy()
        # flip each normal if z component is negative
        face_normals[face_normals[:, 2] < 0] *= -1

        var = np.var(face_normals, axis=0).sum()
        return var < T_VAR

    def estimate_leaf_normal(self, trimesh_mesh: trimesh.Trimesh) -> np.ndarray:
        face_normals = trimesh_mesh.face_normals.copy()
        # flip each normal if z component is negative
        face_normals[face_normals[:, 2] < 0] *= -1
        avg_normal = np.mean(face_normals, axis=0)
        return avg_normal / np.linalg.norm(avg_normal)

    def smooth_leaf(self, ms_tmp: pmeshlab.MeshSet, mesh: trimesh.Trimesh, n_smooth: int, alpha_smooth) -> pmeshlab.Mesh:
        avg_normal = self.estimate_leaf_normal(mesh)

        # calculate viewpoint
        viewpoint = avg_normal * 1000

        # preserve boundaries
        unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        boundary_vertices = np.unique(unique_edges.flatten())

        ms_tmp.add_mesh(trimesh_to_pymesh(mesh))
        v_old = ms_tmp.current_mesh().vertex_matrix().copy()

        # depth smooth
        ms_tmp.apply_coord_depth_smoothing(stepsmoothnum=n_smooth, viewpoint=viewpoint)
        v = ms_tmp.current_mesh().vertex_matrix().copy()

        # revert boundary vertices
        v[boundary_vertices] = v_old[boundary_vertices]

        # for vertices that moved down, only move alpha fraction of the distance
        moved_down = v[:, 2] < v_old[:, 2]
        v[moved_down] = v_old[moved_down] + alpha_smooth * (v[moved_down] - v_old[moved_down])

        f = ms_tmp.current_mesh().face_matrix().copy()
        m = pmeshlab.Mesh(vertex_matrix=v, face_matrix=f)

        return m

    def plot_clusterings(self, X, clustering, n_clusters):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        plot_colors = generate_colormap(n_clusters)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', s=10, color=np.array(plot_colors)[clustering])

        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
        ax.set_title('Predicted labels')

        plt.show()
