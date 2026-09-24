import os
import pickle
from datetime import datetime


def log_data_split(data_split, cfg_dataset):
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'default_value')
    log_dir = cfg_dataset['logdir']

    path_run = os.path.join(log_dir, f'run_{slurm_job_id}')
    os.makedirs(path_run)

    path_data_split = os.path.join(path_run, 'data_split.pickle')

    with open(path_data_split, 'wb') as handle:
        pickle.dump(data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)


def log(message):
    time = datetime.now().strftime("[%H:%M:%S]")
    print(f'{time} {message}')
