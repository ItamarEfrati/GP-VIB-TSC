import os
import torch
from collections import defaultdict

import hydra
import pyrootutils
import pandas as pd

from omegaconf import DictConfig

from src.tasks.eval_tsc_task_inception_encoder import evaluate

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/hmnist.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

DATASETS_UEA = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket', 'DuckDuckGeese',
                'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements',
                'HandMovementDirection', 'Handwriting', 'Heartbeat', 'Libras', 'LSST', 'MotorImagery', 'NATOPS',
                'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
                'StandWalkJump', 'UWaveGestureLibrary', 'EigenWorms']


@hydra.main(version_base="1.2", config_path=os.path.join(root, "configs"), config_name="uea.yaml")
def main(cfg: DictConfig) -> float:
    torch.set_float32_matmul_precision('high')
    num_runs = cfg.num_runs
    seeds = [int.from_bytes(os.urandom(4), byteorder='little', signed=False) for _ in range(num_runs)]

    datasets = DATASETS_UEA
    if cfg.dataset_name:
        datasets = [cfg.dataset_name]

    for dataset_name in datasets:
        dataset_metric_dict = defaultdict(list)
        for i, seed in enumerate(seeds):
            temp_conf = cfg.copy()
            temp_conf['seed'] = seed
            run_dict = evaluate(temp_conf, dataset_name)
            dataset_metric_dict['dataset'].append(dataset_name)

            for k, v in run_dict.items():
                dataset_metric_dict[k].append(float(v))
            df = pd.DataFrame.from_dict(dataset_metric_dict).set_index(['seed', 'dataset'])
            median = df.groupby('dataset').median()
            median.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('median', x), median.index)))
            mean = df.groupby('dataset').mean()
            mean.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('mean', x), mean.index)))
            std = df.groupby('dataset').std()
            std.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('std', x), std.index)))
            df = pd.concat([df, median], axis=0)
            df = pd.concat([df, mean], axis=0)
            df = pd.concat([df, std], axis=0)
            df.index.names = ['seed', 'dataset']
            df.to_csv(os.path.join(cfg.paths.output_dir, f'results_{dataset_name}.csv'))


if __name__ == "__main__":
    main()
