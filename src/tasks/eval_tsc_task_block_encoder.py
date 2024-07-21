import os
from typing import List

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from lightning.fabric.utilities.seed import seed_everything, reset_seed
from lightning.pytorch.loggers import Logger

from src import utils

log = utils.get_pylogger(__name__)


def _init_model(model, datamodule):
    args = {
        "kernel": [3, 3, 3, 3],
        "channels": [32, 16, 8, 8],
        "dropout": [10, 10, 10, 10],
        "hidden": [-1, -1, -1, -1]
    }
    model.keywords['num_classes'] = datamodule.output_size

    # if datamodule.time_series_size <= 50:
    #     start = 1
    # else:
    #     start = 0
    start = 0

    model.keywords['timeseries_encoder'].keywords["timeseries_size"] = datamodule.time_series_size
    model.keywords['timeseries_encoder'].keywords["ts_embedding_size"] = datamodule.channels
    model.keywords['timeseries_encoder'].keywords["n_cnn_layers"] = 4 - start
    for i, j in enumerate(range(start, 4)):
        model.keywords['timeseries_encoder'].keywords[f'out_channels_{i + 1}'] = args['channels'][j]
        model.keywords['timeseries_encoder'].keywords[f'kernel_size_{i + 1}'] = args['kernel'][j]
        model.keywords['timeseries_encoder'].keywords[f'dropout_{i + 1}'] = args['dropout'][j]

        if i <= 2:
            model.keywords['timeseries_encoder'].keywords[f'hidden_size_{i + 1}'] = args['hidden'][j]

    model.keywords['timeseries_encoder'] = model.keywords['timeseries_encoder']()
    model.keywords['decoder'].keywords['output_size'] = datamodule.output_size

    model = model()
    model.class_weight = torch.tensor(datamodule.class_weight, dtype=torch.float)

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
    if config.get("seed"):
        seed = config.seed
        seed_everything(config.seed, workers=True)
    else:
        rand_bytes = os.urandom(4)
        # config['seed'] = int.from_bytes(rand_bytes, byteorder='little', signed=False)
        seed = int.from_bytes(rand_bytes, byteorder='little', signed=False)
        seed_everything(seed, workers=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}> with dataset {dataset_name}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule = datamodule(dataset_name=dataset_name)
    datamodule.prepare_data()

    log.info("Instantiating callbacks...")
    callbacks: List[pl.Callback] = utils.instantiate_callbacks(config.get("callbacks"))

    monitor = 'val_BinaryAccuracy' if datamodule.output_size == 2 else 'val_MulticlassAccuracy'
    temp_callbacks = []
    for c in callbacks:
        try:
            temp_callbacks.append(c(monitor=monitor))
        except:
            temp_callbacks.append(c)
    callbacks = temp_callbacks
    config.model.update({'monitor_metric': monitor})

    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    model = _init_model(model, datamodule)

    if os.name != 'nt':
        torch.compile(model)

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




