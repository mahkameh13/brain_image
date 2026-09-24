import os


def update_logger(logger, cfg):
    _update_tag_list(logger, cfg)


def _update_tag_list(logger, cfg):
    module_model = cfg['model']['_target_']
    name_model = module_model.split('.')[-1]

    path_dataset = cfg['dataset']['data']['root']
    name_dataset = path_dataset.split('/')[4]
    name_modality = cfg['dataset']['data']['mode']

    tags = [name_dataset, name_modality, name_model]
    logger.experiment.tags = tags
    logger.config = cfg
