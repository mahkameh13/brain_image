import os

import numpy as np
import pandas as pd

import latent_mri.utils as utils
from latent_mri.data.datasets import MRIDataset
from latent_mri.dimensionality import Dimensionality


def main():
    path_root = '/scratch/dldevel/galella/IXI/derivatives/mri_crop'
    assert os.path.exists(path_root)

    utils.log('Creating new dimensionality file...\n')
    path_csv = os.path.join('/home/dldevel/galella/LatentSpaceMRI',
                            'results',
                            'dimensionality_IXI.csv')

    if os.path.exists(path_csv):
        df_dims = pd.read_csv(path_csv)
        utils.log('Dimensionalities files found '
                  f'({len(df_dims)} datasets)...\n')
    else:
        utils.log('Creating new dimensionality file...\n')
        columns = ['dataset', 'mode', 'ED', 'ID', 'type', 'images', 'size',
                   'channels']
        df_dims = pd.DataFrame(columns=columns)

    for batch_size in [48, 96, 144, 192, 288]:
        for mode in ['T1', 'T2', 'PD']:
            utils.log(f'Processing {mode}...')

            dataset = MRIDataset(root=path_root,
                                 mode=mode)

            num_images = len(dataset)
            num_voxels_x = dataset[0][0].shape[1]
            num_voxels_y = dataset[0][0].shape[2]
            num_voxels_z = dataset[0][0].shape[3]

            data = np.zeros((num_images,
                             num_voxels_x, num_voxels_y, num_voxels_z))

            utils.log(f'Dataset Size: {num_images}x{num_voxels_x}x{num_voxels_y}x{num_voxels_z}...')
            for i_image in range(num_images):
                data[i_image] = dataset[i_image][0].squeeze(0).cpu().numpy()

            utils.log('Calculating effective dimensionality '
                      f'(batch size {batch_size})...')
            effective_dim = Dimensionality.effective_dimensionality_batch(data, batch_size=batch_size)

            utils.log('Calculating intrinsic dimensionality...')
            intrinsic_dim = Dimensionality.intrinsic_dimensionality(data)

            utils.log(f'ED: {effective_dim:.2f}  ID: {intrinsic_dim:.2f}\n')

            new_row = pd.DataFrame({
                'dataset': 'IXI',
                'mode': [mode],
                'ED': [effective_dim],
                'ID': [intrinsic_dim],
                'type': ['medical'],
                'images': [num_images],
                'size': [num_voxels_x * num_voxels_y * num_voxels_z],
                'channels': [1],
                'batch_size': batch_size
            })

            df_dims = pd.concat([df_dims, new_row], ignore_index=True)
            df_dims.to_csv(path_csv, index=False)


if __name__ == '__main__':
    main()
