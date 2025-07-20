import argparse
import json
import os
import re
import zipfile


def main():
    '''Assumes dataset is a .zip created by BlenderNerf'''

    parser = argparse.ArgumentParser(description='Fix dataset')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('--output', type=str, default='dataset', help='Output directory')
    args = parser.parse_args()
    ds = args.dataset
    output = args.output

    with zipfile.ZipFile(ds, 'r') as zip_ref:
        zip_ref.extractall(args.output)

    # 70-20-10 train-val-test split

    with open(f'{output}/transforms_train.json', 'r+') as f:
        train_json = json.load(f)
        frames = train_json['frames']
        meta = {'camera_angle_x': train_json['camera_angle_x']}

        for frame in frames:
            frame['file_path'] = re.sub(r'\\(\d+)\.png', r'/\1', frame['file_path'])

        no_samples = len(frames)
        val_split = int(no_samples * 0.7)
        test_split = int(no_samples * 0.9)
        print(f'Train: {no_samples}, Test: {test_split}, Val: {val_split}')

        train_json = {**meta, 'frames': frames[:val_split]}
        val_json = {**meta, 'frames': frames[val_split:test_split]}
        test_json = {**meta, 'frames': frames[test_split:]}

    with open(f'{output}/transforms_train.json', 'w') as f:
        json.dump(train_json, f)

    with open(f'{output}/transforms_val.json', 'w') as f:
        json.dump(val_json, f)

    with open(f'{output}/transforms_test.json', 'w') as f:
        json.dump(test_json, f)


if __name__ == '__main__':
    main()
