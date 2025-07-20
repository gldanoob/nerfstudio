import argparse
import glob
import os
import pathlib
import time
import traceback
from typing import Literal

import eval_mesh
import pipeline
import pymeshlab
import tabulate

MESH_FILE = 'poisson_mesh.ply'

# First options are defaults
options = {
    'cleaning': ['z-score'],
    'clustering': ['sample'],
    'smoothing': ['depth'],
    'post_processing': ['close_holes', 'alpha_wrap'],
    'all': ['default']
}

STOP_AT_STAGE: Literal['cleaning', 'clustering', 'smoothing', 'post_processing', 'all'] = 'all'

datasets = [
    'money_no_vase',
    # 'tree_no_vase',
    # 'palm',
    # 'palm2',
    # 'bush',
    # 'agapanthus',
    # 'grass'
]


def fmt_dist(f: float):
    return f'{f*1e4:.4f}'

def fmt_time(t: float):
    return f'{t:.2f}'


def to_file_name(s: str):
    return s.replace(' ', '_').lower()


def main():
    testing_stage_options = options[STOP_AT_STAGE] if STOP_AT_STAGE in options else ['default']

    gt_to_mesh_results = {
        'Method': ['Original'] + [testing_option for testing_option in testing_stage_options],
    }

    mesh_to_gt_results = {
        'Method': ['Original'] + [testing_option for testing_option in testing_stage_options],
    }

    runtime_results = {
        'Method': [testing_option for testing_option in testing_stage_options],
    }

    parser = argparse.ArgumentParser(description='Run methods on coarse mesh')
    parser.add_argument('mesh_dir', type=str, help='Path to coarse mesh directory')
    parser.add_argument('gt_mesh', type=str, help='Path to ground truth mesh')
    args = parser.parse_args()

    mesh_base_dir = pathlib.Path(args.mesh_dir)
    gt_base_dir = pathlib.Path(args.gt_mesh)

    files = glob.glob('tmp/*')
    for f in files:
        os.remove(f)

    for dataset in datasets:
        mesh_dir = mesh_base_dir / dataset
        mesh_path = mesh_dir / MESH_FILE
        gt_path = gt_base_dir / dataset / 'gt.ply'

        ms = pymeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(mesh_path.as_posix())
        m = ms.current_mesh()

        print('=== Mesh Statistics ===')
        print(m.face_number(), 'faces')
        print(m.vertex_number(), 'vertices')
        print('=======================')

        # compare poisson mesh with gt
        print('\n\n\n---> Comparing poisson mesh (target) with ground truth (source)')
        gt_to_mesh_original = eval_mesh.hausdorff_distance(gt_path, mesh_path)
        print(gt_to_mesh_original["mean"])
        gt_to_mesh_results[dataset] = [fmt_dist(gt_to_mesh_original['mean'])]

        print('\n\n\n---> Comparing ground truth (target) with poisson mesh (source)')
        mesh_to_gt_original = eval_mesh.hausdorff_distance(mesh_path, gt_path)
        print(mesh_to_gt_original["mean"])
        mesh_to_gt_results[dataset] = [fmt_dist(mesh_to_gt_original['mean'])]

        runtime_results[dataset] = []

        for testing_option in testing_stage_options:
            try:
                print(f'\n\n\nRunning {testing_option}: stopping after {STOP_AT_STAGE} stage')
                out_path = mesh_dir / f'{to_file_name(testing_option)}.ply'

                args = {
                    stage: testing_option if stage == STOP_AT_STAGE else options[stage][0]
                        for stage in options
                }

                print(args)

                start = time.time()
                pipeline.Pipeline(mesh_path, out_path, gt_path.parent).run(**args, stop_at=STOP_AT_STAGE)
                # methods_good.ClusteringSKLearn(mesh_path, out_path).run()
                end = time.time()
                print(f'Finished in {end - start:.2f} seconds\n')

                # Compare mesh with gt
                print(f'---> Comparing {testing_option} mesh (target) with ground truth (source)')
                gt_to_mesh = eval_mesh.hausdorff_distance(gt_path, out_path)
                print(gt_to_mesh['mean'])

                if gt_to_mesh['mean'] < gt_to_mesh_original['mean']:
                    print(f'---> Improvement: {gt_to_mesh_original["mean"] - gt_to_mesh["mean"]}')
                    gt_to_mesh_results[dataset].append(f'{fmt_dist(gt_to_mesh["mean"])} (improved)')

                else:
                    gt_to_mesh_results[dataset].append(fmt_dist(gt_to_mesh['mean']))

                # Compare gt with mesh
                print(f'---> Comparing ground truth (target) with {testing_option} mesh (source)')
                mesh_to_gt = eval_mesh.hausdorff_distance(out_path, gt_path)
                print(mesh_to_gt['mean'])

                if mesh_to_gt['mean'] < mesh_to_gt_original['mean']:
                    print(f'---> Improvement: {mesh_to_gt_original["mean"] - mesh_to_gt["mean"]}')
                    mesh_to_gt_results[dataset].append(f'{fmt_dist(mesh_to_gt["mean"])} (improved)')

                else:
                    mesh_to_gt_results[dataset].append(fmt_dist(mesh_to_gt['mean']))

                runtime_results[dataset].append(fmt_time(end - start))

            except Exception as e:
                print(f'Error running {testing_option}')
                print(traceback.format_exc())
                gt_to_mesh_results[dataset].append('Error')
                mesh_to_gt_results[dataset].append('Error')
                runtime_results[dataset].append('Error')

    print('\n\n\nGT --> Mesh (10^-4):')
    print(tabulate.tabulate(gt_to_mesh_results, headers='keys', tablefmt='pretty'))

    print('\n\n\nMesh --> GT (10^-4):')
    print(tabulate.tabulate(mesh_to_gt_results, headers='keys', tablefmt='pretty'))

    print('\n\n\nRuntime (s):')
    print(tabulate.tabulate(runtime_results, headers='keys', tablefmt='pretty'))


if __name__ == '__main__':
    main()
