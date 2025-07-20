import copy
import math
import os
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import pymeshlab
import pymeshlab.pmeshlab
import pymeshlab.pmeshlab as pmeshlab
import trimesh
from matplotlib import cm
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.kernel_approximation import pairwise_kernels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

ALPHA_WRAP_ALPHA = 1/200
ALPHA_WRAP_OFFSET = 1/50000

# Reduce number of faces by this factor
DECIMATION_FACTOR = 0.8
# Remove clusters with less than n triangles
REMOVE_CLUSTERS_WITH_FEW_TRIANGLES = 100
# Remove n small clusters
REMOVE_SMALL_CLUSTERS = 20

N_CLUSTERS = 170
LEAF_VARIANCE_THRESHOLD = 0.55
HOLE_SIZE = 100

SPLIT_CLUSTERS = True

SMOOTHING = 20

np.random.seed(RANDOM_STATE)


class Method:
    __name__ = "Method"

    def __init__(self, in_path: Union[pathlib.Path, str], out_path: Union[pathlib.Path, str], tmp_path=None):
        self.in_path = in_path.as_posix() if isinstance(in_path, pathlib.Path) else in_path
        self.out_path = out_path.as_posix() if isinstance(out_path, pathlib.Path) else out_path
        self.tmp_path = 'tmp/tmp.ply'
        pass

    def run(self):
        pass


def remove_small_clusters(mesh, n_clusters):

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print("# of clusters:", len(cluster_n_triangles))

    mesh_0 = copy.deepcopy(mesh)
    mesh_removed = copy.deepcopy(mesh)
    clusters_to_remove = np.argsort(cluster_area)[:n_clusters]
    triangles_to_remove = np.isin(triangle_clusters, clusters_to_remove)

    print('# Triangles to remove:', triangles_to_remove.sum())
    mesh_0.remove_triangles_by_mask(triangles_to_remove)

    mesh_removed.remove_triangles_by_mask(np.logical_not(triangles_to_remove))
    return mesh_0, mesh_removed


def remove_clusters_with_few_triangles(mesh, n_triangles):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print("# of clusters:", len(cluster_n_triangles))

    mesh_0 = copy.deepcopy(mesh)
    mesh_removed = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_triangles

    print('# Triangles to remove:', triangles_to_remove.sum())
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_removed.remove_triangles_by_mask(np.logical_not(triangles_to_remove))
    return mesh_0, mesh_removed


def custom_draw_geometry(mesh, **kwargs):
    mesh_ = copy.deepcopy(mesh)
    mesh_.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_], **kwargs)


class Pipeline(Method):
    __name__ = "Full Pipeline"

    def run(self):
        mesh = o3d.io.read_triangle_mesh(self.in_path)

        ms = pmeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(self.in_path)
        m = ms.current_mesh()

        print('=== Mesh Statistics ===')
        print(m.face_number(), 'faces')
        print(m.vertex_number(), 'vertices')
        print('=======================')

        # Decimate the mesh
        target_faces = int(m.face_number() * DECIMATION_FACTOR)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preserveboundary=True)

        alpha = ALPHA_WRAP_ALPHA
        offset = ALPHA_WRAP_OFFSET
        ms.generate_alpha_wrap(alpha_fraction=alpha, offset_fraction=offset)
        ms.save_current_mesh(self.tmp_path)

        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(self.tmp_path)
        # mesh = mesh.filter_smooth_simple(number_of_iterations=1)

        # v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # v, f = pymeshfix.clean_from_arrays(v, f)

        # mesh.vertices = o3d.utility.Vector3dVector(v)
        # mesh.triangles = o3d.utility.Vector3iVector(f)

        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes(0.001).to_legacy()

        mesh_0, mesh_removed = remove_clusters_with_few_triangles(mesh, REMOVE_CLUSTERS_WITH_FEW_TRIANGLES)

        # save the mesh
        o3d.io.write_triangle_mesh(self.out_path, mesh_0)


class AlphaWrap(Method):
    __name__ = "Alpha Wrap"
    # only apply alpha wrap

    def run(self):
        ms = pmeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(self.in_path)
        m = ms.current_mesh()

        print('=== Mesh Statistics ===')
        print(m.face_number(), 'faces')
        print(m.vertex_number(), 'vertices')
        print('=======================')

        ms.generate_alpha_wrap(alpha_fraction=ALPHA_WRAP_ALPHA, offset_fraction=ALPHA_WRAP_OFFSET)
        ms.save_current_mesh(self.out_path)


class Decimation(Method):
    __name__ = "Decimation"
    # only apply decimation

    def run(self):
        ms = pmeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(self.in_path)

        m = ms.current_mesh()
        target_faces = int(m.face_number() * DECIMATION_FACTOR)

        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preserveboundary=True)
        ms.save_current_mesh(self.out_path)


class SmallClusters1(Method):
    __name__ = "Remove Clusters with Few Triangles"
    # only apply remove clusters with few triangles

    def run(self):
        mesh = o3d.io.read_triangle_mesh(self.in_path)
        mesh_0, mesh_removed = remove_clusters_with_few_triangles(mesh, REMOVE_CLUSTERS_WITH_FEW_TRIANGLES)
        o3d.io.write_triangle_mesh(self.out_path, mesh_0)


class SmallClusters2(Method):
    __name__ = "Remove clusters with small area"
    # only apply remove small clusters

    def run(self):
        mesh = o3d.io.read_triangle_mesh(self.in_path)
        mesh_0, mesh_removed = remove_small_clusters(mesh, REMOVE_SMALL_CLUSTERS)
        o3d.io.write_triangle_mesh(self.out_path, mesh_0)


class CloseHoles(Method):
    __name__ = "Close Holes"

    def run(self):
        ms = pmeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(self.in_path)
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_close_holes(maxholesize=HOLE_SIZE, newfaceselected=False, refinehole=True)
        ms.save_current_mesh(self.out_path)


def plot_vertices(vertices):
    assert vertices.shape[1] == 3
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    plt.show()


class Clustering(Method):
    def run(self):
        mesh = o3d.io.read_triangle_mesh(self.in_path)

        # Decimate the mesh
        # target_faces = 100000
        # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

        # get vertices
        X = np.asarray(mesh.vertices)
        print(X.shape)

        sample = np.random.choice(X.shape[0], 10000, replace=False)
        X = X[sample]
        print(X.shape)

        sigma = 1
        A = -1 * np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1)
        A = np.exp(A / (2 * sigma**2))
        np.fill_diagonal(A, 0)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # bin_values, _, _ = ax1.hist(A.flatten(), bins='auto')

        # # limit the y-axis to the value of the 10th bin
        # ax2.hist(A.flatten(), bins='auto')
        # ax2.set_ylim([0, 100000])

        # plt.show()

        A1 = np.copy(A)
        A1[A1 < 0.9] = 0

        # identity matrix
        I = np.zeros_like(A)
        np.fill_diagonal(I, 1)

        # degree matrix
        D = np.zeros_like(A)
        np.fill_diagonal(D, np.sum(A, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))

        L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)

        print('Computing eigenvectors...')
        eigenvalues, eigenvectors = np.linalg.eig(L)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        print('Sorting eigenvectors...')
        # Order the eigenvalues in an increasing order
        ind = np.argsort(eigenvalues, axis=0)
        eigenvalues_sorted = np.take_along_axis(eigenvalues, ind, axis=0)

        # Order the eigenvectors based on the magnitude of their corresponding eigenvalues
        eigenvectors_sorted = eigenvectors.take(ind, axis=1)

        # fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        # eigen_v_x = np.linspace(0, eigenvectors_sorted.shape[0], eigenvectors_sorted.shape[0])

        # for j, ax in enumerate(fig.axes):
        #     eigen_v_y = eigenvectors_sorted[:, j]
        #     ax.scatter(eigen_v_x, eigen_v_y, marker='o')
        #     ax.set_title(f'eigenvector {j} | eigenvalue: {eigenvalues_sorted[j]:.4f}')

        # plt.show()

        print('Clustering...')
        X_transformed = eigenvectors_sorted[:, [0, 3]]

        scaler = StandardScaler()
        scaler.fit(X_transformed)
        X_transformed_scaled = scaler.transform(X_transformed)

        kmeans = KMeans(n_clusters=20, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(X_transformed_scaled)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        plot_colors = cm.tab20.colors
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', s=40, color=np.array(plot_colors)[kmeans.labels_])
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.set(xlabel=None, ylabel=None)
        ax.set_title('Predicted labels')

        plt.show()

        exit()


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
    return trimesh.Trimesh(v, f, process=False)


def trimesh_to_pymesh(mesh: trimesh.Trimesh):
    v = mesh.vertices
    f = mesh.faces
    m = pmeshlab.Mesh(vertex_matrix=v, face_matrix=f)
    return m


class ClusteringSKLearn(Method):
    def run(self):
        # Remove small clusters
        # TODO: remove clusters far from the center (DBSCAN?)
        # SmallClusters1(self.in_path, self.tmp_path).run()

        ms = pmeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(self.in_path)
        m = ms.current_mesh()

        # Decimate the mesh if more than 1M faces
        # if m.face_number() > 1000000:
        #     ms.meshing_decimation_quadric_edge_collapse(targetfacenum=1000000, preserveboundary=True)

        # sample vertices
        X = np.asarray(m.vertex_matrix())
        sample = np.random.choice(X.shape[0], 10000, replace=False)
        X = X[sample]
        print('Sampled', X.shape[0], 'vertices for clustering')

        # spectral clustering on sample
        affinity = pairwise_kernels(X, metric='rbf', gamma=30)
        clustering = spectral_clustering(
            affinity=affinity, n_clusters=N_CLUSTERS, assign_labels="cluster_qr", random_state=RANDOM_STATE
        )

        # KNN on the rest of the vertices
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(X, clustering)

        X_all = np.asarray(m.vertex_matrix())
        print('Clustering the rest of the vertices')
        clustering_all = classifier.predict(X_all)

        # self.plot_clusterings(X_all, clustering_all, N_CLUSTERS)

        # get submesh for each cluster
        trimesh_mesh = pymesh_to_trimesh(m)
        subfaces_list = []

        for i in range(N_CLUSTERS):
            print('Getting cluster #', i)
            no_vertices = np.sum(clustering_all == i)
            if no_vertices < 3:
                print('Skipping cluster with less than 3 vertices')
                continue

            print('Number of vertices:', no_vertices)

            cluster_indices = np.where(clustering_all == i)[0]
            subfaces = np.any(np.isin(trimesh_mesh.faces, cluster_indices), axis=1)
            subfaces_list.append(subfaces)

        # subfaces_list.sort(key=lambda x: x.sum(), reverse=True)
        cluster_meshes = trimesh_mesh.submesh(subfaces_list, append=False)

        # process each submesh
        ms_new = pmeshlab.MeshSet(verbose=True)
        ms_tmp = pmeshlab.MeshSet(verbose=True)
        for i, cluster_mesh in enumerate(cluster_meshes):
            if SPLIT_CLUSTERS:
                # split submesh into connected components
                # TODO: optimize
                print('Splitting cluster #', i)
                labels = trimesh.graph.connected_component_labels(cluster_mesh.face_adjacency)
                print('Number of components:', labels.max() + 1)
                subcluster_faces_list = []
                for j in range(labels.max() + 1):
                    subfaces = labels == j
                    subcluster_faces_list.append(subfaces)

                subcluster_faces_list.sort(key=lambda x: x.sum(), reverse=True)
                subcluster_meshes = cluster_mesh.submesh(subcluster_faces_list, append=False)

                for j, subcluster_mesh in enumerate(subcluster_meshes):
                    # add submesh to new meshset
                    if j < 10 and self.is_leaf_heuristic(subcluster_mesh):
                        subcluster_mesh.export(f'tmp/cluster_{i}_{j}.ply')
                        print('Smoothing leaf cluster #', i)
                        smoothed = self.smooth_leaf(ms_tmp, subcluster_mesh)
                        ms_new.add_mesh(smoothed)
                    else:
                        ms_new.add_mesh(trimesh_to_pymesh(subcluster_mesh))

            else:
                # add submesh to new meshset
                ms_tmp.add_mesh(trimesh_to_pymesh(cluster_mesh))
                if self.is_leaf_heuristic(cluster_mesh):
                    cluster_mesh.export(f'tmp/cluster_{i}.ply')
                    print('Smoothing leaf cluster')
                    smoothed = self.smooth_leaf(ms_tmp, cluster_mesh)
                    ms_new.add_mesh(smoothed)
                else:
                    ms_new.add_mesh(trimesh_to_pymesh(cluster_mesh))

        # combine all submeshes
        ms_new.generate_by_merging_visible_meshes()

        # run close holes method
        ms_new.meshing_repair_non_manifold_edges()
        ms_new.meshing_close_holes(maxholesize=HOLE_SIZE, newfaceselected=False, refinehole=True)
        ms_new.save_current_mesh(self.out_path)

    def is_leaf_heuristic(self, mesh: trimesh.Trimesh) -> bool:
        # TODO: test this heuristic
        if mesh.faces.shape[0] < 100:
            return False

        face_normals = mesh.face_normals.copy()
        # flip each normal if z component is negative
        face_normals[face_normals[:, 2] < 0] *= -1

        var = np.var(face_normals, axis=0).sum()
        # more than 100 triangles and low variance
        return var < LEAF_VARIANCE_THRESHOLD

    def estimate_leaf_normal(self, trimesh_mesh: trimesh.Trimesh) -> np.ndarray:
        face_normals = trimesh_mesh.face_normals.copy()
        # flip each normal if z component is negative
        face_normals[face_normals[:, 2] < 0] *= -1
        avg_normal = np.mean(face_normals, axis=0)
        return avg_normal / np.linalg.norm(avg_normal)

    def smooth_leaf(self, ms_tmp: pmeshlab.MeshSet, mesh: trimesh.Trimesh) -> pmeshlab.Mesh:
        avg_normal = self.estimate_leaf_normal(mesh)

        # calculate viewpoint
        viewpoint = avg_normal * 1000

        # depth smooth
        # TODO: preserve boundaries
        ms_tmp.add_mesh(trimesh_to_pymesh(mesh))
        v_old = ms_tmp.current_mesh().vertex_matrix().copy()

        ms_tmp.apply_coord_depth_smoothing(stepsmoothnum=SMOOTHING, viewpoint=viewpoint)
        v = ms_tmp.current_mesh().vertex_matrix().copy()

        # only save vertices that have moved up
        moved_down = v[:, 2] < v_old[:, 2]
        v[moved_down] = v_old[moved_down]

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

