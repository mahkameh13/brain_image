import os

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import Callback


class LogDatasetInformation(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        path_dataset = \
            trainer.datamodule.train_set.dataset.root.split('/derivatives')[0]
        name_dataset = os.path.basename(path_dataset)
        path_dataset_info = os.path.join(path_dataset, 'participants.tsv')
        df_data_info = pd.read_csv(path_dataset_info, sep='\t')
        df_data_info['split'] = None

        data_split = trainer.datamodule.data_split
        idxs_train, labels_train = data_split['train']
        idxs_valid, labels_valid = data_split['valid']
        idxs_test, labels_test = data_split['test']

        num_subjects = len(labels_train) + len(labels_valid) + len(labels_test)

        for i_subject in range(num_subjects):
            if i_subject in idxs_train:
                label = labels_train[idxs_train.index(i_subject)]
                split = 'Train'
            elif i_subject in idxs_valid:
                label = labels_valid[idxs_valid.index(i_subject)]
                split = 'Valid'
            elif i_subject in idxs_test:
                label = labels_test[idxs_test.index(i_subject)]
                split = 'Test'
            else:
                raise Exception(f'Subject {i_subject} not Test')

            condition = df_data_info['participant_id'] == label
            df_data_info.loc[condition, 'split'] = split

        pl_module.logger.log_table(key=name_dataset,
                                   dataframe=df_data_info)


class LogDatasetSplits(Callback):
    def __init__(self, num_images):
        self.num_images = num_images

    def on_train_start(self, trainer, pl_module) -> None:
        datasets = {'Train': trainer.datamodule.train_set,
                    'Valid': trainer.datamodule.valid_set,
                    'Test': trainer.datamodule.test_set}

        for name, dataset in datasets.items():
            images = []
            caption = []

            for idx in range(self.num_images):
                image, label = dataset[idx]
                images.append(np.rot90(image.numpy()[0, :, :, 95]))
                caption.append(label)

            pl_module.logger.log_image(key=name,
                                       images=images,
                                       caption=caption)


class LogReconstructedInputs(Callback):
    def __init__(self, num_images, num_epochs, use_sigmoid=True):
        super().__init__()
        self.num_images = num_images
        self.num_epochs = num_epochs
        self.use_sigmoid = use_sigmoid
        wandb.define_metric('Epoch')
        wandb.define_metric('Valid Reconstructions', step_metric='Epoch')

    def _get_images(self, x_recon, labels):
        subset_predictions = x_recon[:self.num_images]
        subset_labels = labels[:self.num_images]

        if self.use_sigmoid:
            subset_predictions = F.sigmoid(subset_predictions)

        subset_predictions = subset_predictions.detach().cpu().double().squeeze(0).numpy()
        images = [np.rot90(image[0, :, :, 95]) for image in subset_predictions]
        caption = subset_labels[:self.num_images]

        return images, caption

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        is_sanity_check = trainer.sanity_checking
        validation_dict = pl_module.validation_step_dict
        current_epoch = pl_module.current_epoch

        if not is_sanity_check and current_epoch % self.num_epochs == 0:
            x_recon = validation_dict['x_recon']
            labels = validation_dict['labels']

            images, caption = self._get_images(x_recon, labels)
            pl_module.logger.log_image(key='Valid Reconstructions',
                                       images=images,
                                       caption=caption,
                                       step=current_epoch)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        test_dict = pl_module.test_step_dict

        x_recon = test_dict['x_recon']
        labels = test_dict['labels']

        images, caption = self._get_images(x_recon, labels)
        pl_module.logger.log_image(key='Test Reconstructions',
                                       images=images,
                                       caption=caption)


class LogMetrics(Callback):
    def __init__(self, num_epochs, metrics):
        super().__init__()
        self.num_epochs = num_epochs
        self.metrics = metrics

    def _run_metrics(self, x, x_recon, label):
        dict_metrics = {}

        for metric in self.metrics:
            name_metric = f'{label}_{metric.__class__.__name__}'
            dict_metrics[name_metric] = metric(x, x_recon)

        return dict_metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        is_sanity_check = trainer.sanity_checking
        validation_dict = pl_module.validation_step_dict
        current_epoch = pl_module.current_epoch

        if not is_sanity_check and current_epoch % self.num_epochs == 0:
            x = validation_dict['x']
            x_recon = validation_dict['x_recon']
            x_recon = x_recon.to(dtype=x.dtype)

            batch_size = x.shape[0]

            dict_metrics = self._run_metrics(x, x_recon, 'valid')

            pl_module.log_dict(dict_metrics, batch_size=batch_size,
                               on_step=False, on_epoch=True, 
                               prog_bar=True, logger=True,
                               sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        test_dict = pl_module.test_step_dict

        x = test_dict['x']
        x_recon = test_dict['x_recon']
        x_recon = x_recon.to(dtype=x.dtype)

        batch_size = x.shape[0]

        dict_metrics = self._run_metrics(x, x_recon, 'test')

        pl_module.log_dict(dict_metrics, batch_size=batch_size,
                           on_step=False, on_epoch=True, 
                           prog_bar=True, logger=True,
                           sync_dist=True)


"""
class LogInterpolatedInputs(Callback):
    def __init__(self, num_images, num_epochs):
        super().__init__()
        self.num_images = num_images
        self.num_epochs = num_epochs
        wandb.define_metric('Epoch')

    def _get_embeddings(self, outputs):
        embeddings = torch.cat([prediction[2] for prediction in outputs])
        labels = [label
                  for prediction in outputs
                  for label in prediction[1]]
        caption = \
            [labels[0]] + list(range(1, self.num_images - 1)) + [labels[1]]
        return embeddings, caption

    def _interpolate_embeddings(self, model, embedding1, embedding2):
        embeddings_interpolated = \
            torch.stack([(t * embedding1 + (1 - t) * embedding2).unsqueeze(0)
                        for t in torch.linspace(0, 1, self.num_images)])
        print(embeddings_interpolated.shape)
        recon_interpolated = \
            model.decode_forward(embeddings_interpolated,
                                 use_sigmoid=True)
        recon_interpolated = \
            recon_interpolated.detach().cpu().double().squeeze(0).numpy()
        recons = [np.rot90(recon[0, :, :, 74]) for recon in recon_interpolated]
        return recons

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        is_sanity_check = trainer.sanity_checking
        validation_outputs = pl_module.validation_step_outputs
        current_epoch = pl_module.current_epoch
        model = pl_module.model

        if not is_sanity_check and current_epoch % self.num_epochs == 0:
            embeddings, caption = self._get_embeddings(validation_outputs)
            recons = self._interpolate_embeddings(model,
                                                  embeddings[0],
                                                  embeddings[1])
            pl_module.logger.log_image(key='Valid Interpolations',
                                       images=recons,
                                       caption=caption,
                                       step=current_epoch)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        test_outputs = pl_module.test_step_outputs
        model = pl_module.model
        embeddings, caption = self._get_embeddings(test_outputs)
        recons = self._interpolate_embeddings(model,
                                              embeddings[0],
                                              embeddings[1])
        pl_module.logger.log_image(key='Test Interpolations',
                                   images=recons,
                                   caption=caption)
"""