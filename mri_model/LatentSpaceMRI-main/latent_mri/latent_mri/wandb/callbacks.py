import os

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import Callback


class LogDatasetInformation(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        list_datasets = trainer.datamodule.list_datasets
        for dataset in list_datasets:
            path_dataset = dataset.path.split('/derivatives')[0]
            name_dataset = os.path.basename(path_dataset)
            labels = dataset.labels
            path_dataset_info = os.path.join(path_dataset, 'participants.tsv')
            df_data_info = pd.read_csv(path_dataset_info, sep='\t')
            df_data_info['split'] = None
            idxs_train, idxs_valid, idxs_test = trainer.datamodule.indices
            for i_label, label in enumerate(labels):
                condition = df_data_info['participant_id'] == label
                if i_label in idxs_train:
                    df_data_info.loc[condition, 'split'] = 'Train'
                elif i_label in idxs_valid:
                    df_data_info.loc[condition, 'split'] = 'Valid'
                elif i_label in idxs_test:
                    df_data_info.loc[condition, 'split'] = 'Test'
                else:
                    raise Exception(f'Subject {i_label} not Test')

            pl_module.logger.log_table(key=name_dataset,
                                       dataframe=df_data_info)


class LogDatasetSplits(Callback):
    def __init__(self, num_images):
        self.num_images = num_images

    def on_train_start(self, trainer, pl_module) -> None:
        datasets = trainer.datamodule.datasets
        list_idxs = trainer.datamodule.indices
        list_keys = ['Train Set', 'Valid Set', 'Test Set']

        for set_idxs, set_key in zip(list_idxs, list_keys):
            idxs = set_idxs[:self.num_images]
            images = []
            caption = []
            for idx in idxs:
                image, label = datasets[idx]
                images.append(image.numpy())
                caption.append(label)

            pl_module.logger.log_image(key=set_key,
                                       images=images,
                                       caption=caption)


class LogReconstructedInputs(Callback):
    def __init__(self, num_images, num_epochs):
        super().__init__()
        self.num_images = num_images
        self.num_epochs = num_epochs
        wandb.define_metric('Epoch')
        wandb.define_metric('Valid Reconstructions', step_metric='Epoch')

    def _get_images(self, outputs):
        predictions = torch.cat([prediction[0] for prediction in outputs])
        labels = [label
                  for prediction in outputs
                  for label in prediction[1]]
        x_recon = F.sigmoid(predictions[:self.num_images])
        x_recon = x_recon.squeeze(0).detach().cpu().double().numpy()
        images = [image for image in x_recon]
        caption = labels[:self.num_images]

        return images, caption

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        is_sanity_check = trainer.sanity_checking
        validation_outputs = pl_module.validation_step_outputs
        current_epoch = pl_module.current_epoch

        if not is_sanity_check and current_epoch % self.num_epochs == 0:
            images, caption = self._get_images(validation_outputs)
            pl_module.logger.log_image(key='Valid Reconstructions',
                                       images=images,
                                       caption=caption,
                                       step=current_epoch)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        test_outputs = pl_module.test_step_outputs
        images, caption = self._get_images(test_outputs)
        pl_module.logger.log_image(key='Test Reconstructions',
                                       images=images,
                                       caption=caption)


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
        recon_interpolated = \
            model.decode_forward(embeddings_interpolated,
                                 use_sigmoid=True)
        recon_interpolated = \
            recon_interpolated.detach().cpu().double().numpy()
        recons = [recon for recon in recon_interpolated]
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


class LogMetrics(Callback):
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
        recon_interpolated = \
            model.decode_forward(embeddings_interpolated,
                                 use_sigmoid=True)
        recon_interpolated = \
            recon_interpolated.detach().cpu().double().numpy()
        recons = [recon for recon in recon_interpolated]
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
def log_embeddings(logger, predictions):
    embeddings = \
        torch.cat([prediction[5] for prediction in predictions])
    num_dims = embeddings.shape[1]
    embeddings = \
        [embedding.tolist()
         for embedding in embeddings.detach().cpu().double().numpy()]

    columns = [f'D{i_dim}' for i_dim in range(num_dims)]
    embedding_table = wandb.Table(columns=columns, data=embeddings)
    wandb.log({'embedding': embedding_table})

"""
