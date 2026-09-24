import hydra
import torch
import latent_mri.wandb.utils as wutils
import latent_mri.data.utils as dutils
from torchinfo import summary
from hydra.utils import get_class


@hydra.main(version_base=None, config_path='config', config_name='autoencoderKL')
def main(cfg) -> None:
    # Check GPU is available.
    if not torch.cuda.is_available():
        raise Exception('GPU not available!')

    # Set matrix multiplication precision (supress warning).
    torch.set_float32_matmul_precision('medium')

    # Create splits and datasets
    dutils.set_data_seed(cfg.dataset)
    partial_dataset = hydra.utils.instantiate(cfg.dataset.data)
    data_splits = dutils.create_data_splits(cfg.dataset)
    transforms = hydra.utils.instantiate(cfg.dataset.transforms)

    # Log data split
    dutils.log_data_split(data_splits, cfg.dataset)

    # Initialize previous existing logger.
    lightning_logger = hydra.utils.instantiate(cfg.lightning_logger)
    wutils.update_logger(lightning_logger, cfg)

    # Set seed for reproducibility.
    hydra.utils.instantiate(cfg.lightning_seed)

    # Load data, create splits and dataloaders.
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.create_datasets(partial_dataset, data_splits, transforms)
    datamodule.print()

    # Set seed (train) for reproducibility.
    hydra.utils.instantiate(cfg.lightning_seed)

    # Load model.
    model = hydra.utils.instantiate(cfg.model)
    input_size = datamodule.train_set[0][0].shape
    summary(model.model, input_size=(1, *input_size))

    # Train model.
    trainer = hydra.utils.instantiate(cfg.trainer, logger=lightning_logger)
    trainer.fit(model, datamodule=datamodule)

    # Load best model and test it.
    last_checkpoint = f'logs/models/last_{cfg["run_id"]}.ckpt'
    trainer.save_checkpoint(last_checkpoint)

    best_valid_checkpoint = f'logs/models/best_model_{cfg["run_id"]}.ckpt'
    model_class = get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(best_valid_checkpoint)
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
