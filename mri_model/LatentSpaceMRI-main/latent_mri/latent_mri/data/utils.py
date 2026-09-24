
import os
import glob
import pickle

import numpy as np
from omegaconf import DictConfig, OmegaConf


def set_data_seed(cfg_dataset):
    seed = cfg_dataset.seed
    print(f'Setting data seed to {seed}')
    np.random.seed(seed)


def create_data_splits(cfg_dataset):
    ratio = cfg_dataset.ratio
    data = cfg_dataset.data
    path_data = data['root']

    labels = get_data_labels(path_data)
    labels = [label for label in labels if label != 'IXI014']
    assert 'IXI014' not in labels
    print('Removing subject IXI014 (low resolution)!')
    num_subjects = len(labels)

    train_ratio, valid_ratio, test_ratio = ratio
    assert train_ratio + valid_ratio + test_ratio == 1

    num_train = round(num_subjects * train_ratio)
    num_valid = round(num_subjects * valid_ratio)
    num_test = num_subjects - num_train - num_valid
    assert num_test > 0

    list_idxs = np.arange(num_subjects)
    np.random.shuffle(list_idxs)

    idxs_train = sorted(list_idxs[:num_train])
    idxs_valid = sorted(list_idxs[num_train:num_train+num_valid])
    idxs_test = sorted(list_idxs[num_train+num_valid:])

    labels = np.array(labels)
    labels_train = labels[idxs_train]
    labels_valid = labels[idxs_valid]
    labels_test = labels[idxs_test]

    data_splits = {
        'train': (idxs_train, labels_train),
        'valid': (idxs_valid, labels_valid),
        'test': (idxs_test, labels_test)
    }

    return data_splits


def get_data_labels(path_data):
    labels = sorted(glob.glob(os.path.join(path_data, '*')))
    return labels


def log_data_split(data_split, cfg_dataset):
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'default_value')
    log_dir = cfg_dataset['logdir']

    path_run = os.path.join(log_dir, f'run_{slurm_job_id}')
    os.makedirs(path_run)

    path_data_split = os.path.join(path_run, 'data_split.pickle')

    with open(path_data_split, 'wb') as handle:
        pickle.dump(data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
