import argparse
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
from queue import Queue
from threading import Thread
from functools import partial
from tqdm import tqdm
import h5py
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superpoint+lightglue': {
        'output': 'matches-superpoint-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'superpoint',
        },
    },
    'disk+lightglue': {
        'output': 'matches-disk-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'disk',
        },
    },
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'adalam': {
        'output': 'matches-adalam',
        'model': {
            'name': 'adalam'
        },
    },
    # roma matcher
    'roma': {
        'output': 'matches-roma',
        'model': {
            'name': 'roma',
            'max_keypoints': 100,
        },
    },
    # lightglue_gim matcher
    'lightglue_gim': {
        'output': 'matches-lightglue_gim',
        'model': {
            'name': 'lightglue_gim',
        },
    },
}


class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r
    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k+'0'] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        with h5py.File(self.feature_path_r, 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k+'1'] = torch.from_numpy(v.__array__()).float()
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        return data
    # def __getitem__(self, idx):
    #     name0, name1 = self.pairs[idx] # name0: name of first element in the pair, name1: name of second element in the pair
    #     data0 = {}
    #     data1 = {}
    #     with h5py.File(self.feature_path_q, 'r') as fd:
    #         grp = fd[name0]
    #         for k, v in grp.items():
    #             try:
    #                 data0[int(k)] = torch.from_numpy(v.__array__()).float()
    #             except:
    #                 data0[k] = torch.from_numpy(v.__array__()).float()
    #         # some matchers might expect an image but only use its size
    #         # data0['image'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
    #     with h5py.File(self.feature_path_r, 'r') as fd:
    #         grp = fd[name1]
    #         for k, v in grp.items():
    #             try:
    #                 data1[int(k)] = torch.from_numpy(v.__array__()).float()
    #             except:
    #                 data1[k] = torch.from_numpy(v.__array__()).float()
    #         # data1['image'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
    #     return data0, data1

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
        print(f"\n-----shape matches: {matches.shape}----\n")
        print("Datasets in group:", list(grp.keys()))
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)
            print(f"\n-----shape scores: {scores.shape}----\n")
            print("Datasets in group:", list(grp.keys()))


def main(conf: Dict,
         pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         overwrite: bool = False) -> Path:

    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features+'.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False) -> Path:
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
    is_roma=False
    if conf['model']['name'] == 'roma':
        is_roma=True
    # fig, ax = plt.subplots(dataset.__len__(), 1)
    for idx, data in enumerate(tqdm(loader, smoothing=.1)):
        
        # print(f"\n--------------len data------{len(data)}\n")
        # data0 = {k: v if k.startswith('image')
        #         else v.to(device, non_blocking=True) for k, v in data[0].items()}
        # data1 = {k: v if k.startswith('image')
        #         else v.to(device, non_blocking=True) for k, v in data[1].items()}
        if is_roma:
            dir_folder_image_test = '/home/vuthanhdat/VisualLocalization/crocodl-benchmark/external/hloc/datasets/South-Building'
            # plot 2 images in the pair
            fig, ax = plt.subplots(1, 2)
            name_image0 = dataset.pairs[idx][0]
            name_image1 = dataset.pairs[idx][1]
            dir_image0 = dir_folder_image_test + '/images/' + name_image0
            dir_image1 = dir_folder_image_test + '/images/' + name_image1
            image0 = cv2.imread(str(dir_image0))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image1 = cv2.imread(str(dir_image1))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            h, w = data[0][1].shape[-2:]
            image0 = cv2.resize(image0, (w, h))
            h, w = data[1][1].shape[-2:]
            image1 = cv2.resize(image1, (w, h))
            pred = model(data)
            # print(f"\n----------shape keypoints0 : {pred['keypoints0'].shape} \n shape keypoints1 : {pred['keypoints1'].shape} \n shape mconf : {pred['mconf'].shape}--------\n")
            # print(f"\n----------keypoints0 {pred['keypoints0']} --------\n")
            # print(f"\n----------keypoints1 {pred['keypoints1']} --------\n")
            # print(f"\n----------mconf {pred['mconf']} --------\n")
            mconf_list = list(pred['mconf'].cpu().numpy())
            # plot range of value in mconf
            keypoints0 = list(pred['keypoints0'].cpu().numpy())
            # print(f"\n----------keypoints0 {keypoints0} --------\n")
            keypoints0 = [[int(point[0]), int(point[1])] for point in keypoints0]
            # print(f"\n----------keypoints0 {keypoints0} --------\n")
            keypoints1 = list(pred['keypoints1'].cpu().numpy())
            keypoints1 = [[int(point[0]), int(point[1])] for point in keypoints1]
            # plot dot point in 2 images
            for i in range(len(keypoints0)):
                # > 0.5 -> green point, else red point
                if mconf_list[i] > 0.5:
                    cv2.circle(image0, (keypoints0[i][0], keypoints0[i][1]), 2, (0, 255, 0), -1)
                else:
                    cv2.circle(image0, (keypoints0[i][0], keypoints0[i][1]), 2, (0, 0, 255), -1)
            for i in range(len(keypoints1)):
                # > 0.5 -> green point, else red point
                if mconf_list[i] > 0.5:
                    cv2.circle(image1, (keypoints1[i][0], keypoints1[i][1]), 2, (0, 255, 0), -1)
                else:
                    cv2.circle(image1, (keypoints1[i][0], keypoints1[i][1]), 2, (0, 0, 255), -1)
            # plot line between two corresponding points
            
            combined = np.hstack((image0, image1))


            ax[0].imshow(combined)

            for i in range(len(keypoints0)):
                p1 = keypoints0[i]
                p2 = keypoints1[i]
                if mconf_list[i] > 0.5:
                    ax[0].plot([p1[0], p2[0] + image0.shape[1]],
                            [p1[1], p2[1]], 
                            color='lime', linewidth=1)

            
            ax[0].set_title('Keypoints and Matches')
            
            value_mconf = pred['mconf'].cpu().numpy()
            ax[1].hist(value_mconf, bins=50)
            ax[1].set_title('Histogram of mconf values')
            ax[1].set_xlabel('Value')
            ax[1].set_ylabel('Frequency')
            plt.show()
            continue
        else:
            print(f"\n--data 0 keys: {data.keys()}---\n")
            data = {k: v if k.startswith('image')
                else v.to(device, non_blocking=True) for k, v in data.items()}
            pred = model(data)
            pair = names_to_pair(*pairs[idx])
            print(f"\n------------ predict dict keys--------{pred.keys()}\n")
            writer_queue.put((pair, pred))
    # return
    writer_queue.join()
    logger.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
