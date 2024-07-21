import os
import hydra
import torch
import lightning.pytorch as pl

from typing import List

from omegaconf import DictConfig
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.loggers import Logger

from src import utils
from utils import datasets_utils

log = utils.get_pylogger(__name__)


def _get_datamodule(config, dataset_name):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule = datamodule(dataset_name=dataset_name)
    datamodule.prepare_data()
    return datamodule


def get_encoding_series_size(train_size):
    if train_size <= 50:
        return 2
    elif train_size <= 200:
        return 3
    elif train_size <= 400:
        return 4
    elif train_size <= 500:
        return 5
    elif train_size <= 700:
        return 6
    elif train_size <= 900:
        return 7
    elif train_size <= 1800:
        return 8
    elif train_size <= 3700:
        return 9
    return 10


def get_encoding_dimension(n_filters):
    return n_filters * 4


def _get_model(config, datamodule):
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    model.keywords['timeseries_encoder'].keywords['input_n_channels'] = datamodule.channels
    n_filters = model.keywords['timeseries_encoder'].keywords['number_of_filters']
    encoding_size = get_encoding_dimension(n_filters)
    time_series_encoding_size = get_encoding_series_size(datamodule.train_size)
    model.keywords['timeseries_encoder'].keywords['encoding_size'] = encoding_size
    model.keywords['timeseries_encoder'].keywords['encoding_series_size'] = time_series_encoding_size
    model.keywords['timeseries_encoder'] = model.keywords['timeseries_encoder']()
    model.keywords['encoder'].keywords['encoding_size'] = encoding_size
    model.keywords['decoder'].keywords['z_dim'] = encoding_size
    model.keywords['decoder'].keywords['output_size'] = datamodule.output_size
    model.keywords['num_classes'] = datamodule.output_size
    model = model(num_classes=datamodule.output_size)
    return model


@utils.task_wrapper
def evaluate(config: DictConfig, *args) -> dict:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        dict: Dict with metrics
    """
    dataset_name = args[0]
    seed = datasets_utils.set_seed(config)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}> with dataset {dataset_name}")
    datamodule = _get_datamodule(config, dataset_name)

    log.info(f"Instantiating model <{config.model._target_}>")
    model = _get_model(config, datamodule)

    if os.name != 'nt':
        torch.compile(model)

    log.info("Instantiating callbacks...")
    callbacks: List[pl.Callback] = utils.instantiate_callbacks(config.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(config.get("logger"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    if logger:
        log.info("Logging hyperparameters!")
        hparams = {
            'seed': seed,
            'config': config.copy(),
            'datamodule': datamodule,
            'model': model,
            'callbacks': callbacks,
            'trainer': trainer,
        }
    utils.log_hyperparameters(hparams=hparams, metrics=dict(config.metrics.metrics))

    ckpt_path = config.get("ckpt_path")

    if ckpt_path is None:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

    metrics_dict = {'seed': config['seed']}

    log.info("Starting testing!")
    reset_seed()
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    metrics_dict.update(trainer.callback_metrics)

    trainer.logger.log_metrics(
        {f'optimize_{config.get("optimized_metric")}': trainer.checkpoint_callback.best_model_score})

    metrics = {}
    for k, v in metrics_dict.items():
        new_key = k.replace('Multiclass', '')
        new_key = new_key.replace('Binary', '')
        metrics[new_key] = v
    return metrics
