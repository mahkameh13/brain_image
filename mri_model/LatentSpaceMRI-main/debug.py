import hydra
import torch
import unimodal_ae.data.utils as dutils


@hydra.main(version_base=None)
def main(cfg) -> None:
    # Check GPU is available.
    if not torch.cuda.is_available():
        raise Exception('GPU not available!')

    # Set matrix multiplication precision (supress warning).
    torch.set_float32_matmul_precision('medium')

    # Create splits and datasets
    dutils.set_data_seed(cfg.dataset)
    partial_dataset = hydra.utils.instantiate(cfg.dataset.data)
    indices_train, indices_valid, indices_test =  \
        dutils.create_data_splits(cfg.dataset)
    transforms_train = hydra.utils.instantiate(cfg.dataset.transforms.train)
    transforms_valid = hydra.utils.instantiate(cfg.dataset.transforms.valid)
    transforms_test = hydra.utils.instantiate(cfg.dataset.transforms.test)

    # Set seed for reproducibility.
    hydra.utils.instantiate(cfg.lightning_seed)

    # Load data, create splits and dataloaders.
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup_train_set(partial_dataset, indices_train, transforms_train)
    datamodule.setup_valid_set(partial_dataset, indices_valid, transforms_valid)
    datamodule.setup_test_set(partial_dataset, indices_test, transforms_test)
    datamodule.print_info()

    # Load model.
    model = hydra.utils.instantiate(cfg.model)

    # Print model info.
    data, _ = datamodule.datasets[0]
    input_size = list(data.unsqueeze(0).size())
    input_size[0] = datamodule.params_dataloaders['train']['batch_size']
    summary(model, input_size=input_size, depth=5)

    # Train model.
    # trainer = hydra.utils.instantiate(cfg.trainer)
    # trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
