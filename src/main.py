import os
import torch
import argparse
from collections import defaultdict

import hydra
import pyrootutils
import pandas as pd

from omegaconf import DictConfig

from src.tasks.eval_tsc_task_inception_encoder import evaluate

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

DATASETS_UCR_2018 = ['ACSF1', 'Adiac', 'ArrowHead', 'BME', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car',
                     'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                     'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
                     'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays',
                     'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'ElectricDevices', 'EthanolLevel',
                     'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                     'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
                     'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty',
                     'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                     'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat',
                     'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                     'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain',
                     'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'OliveOil',
                     'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure',
                     'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock', 'ScreenType',
                     'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
                     'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                     'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                     'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns',
                     'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
                     'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

DATASETS_UEA = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket',
                'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
                'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat',
                'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra',
                'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump',
                'UWaveGestureLibrary']

IS_HYPER_SEARCH = False


@hydra.main(version_base="1.2", config_path=os.path.join(root, "configs"), config_name="config.yaml")
def main(cfg: DictConfig) -> float:
    parser = argparse.ArgumentParser(description="Run GP-VIB on time series datasets.")
    parser.add_argument('--dataset_type', type=str, choices=['UCR', 'UEA'], required=True,
                        help="Type of dataset to run (UCR or UEA).")
    parser.add_argument('--dataset_name', type=str, default=None,
                        help="Name of the specific dataset to run. If not provided, all datasets will be run.")
    parser.add_argument('--num_runs', type=int, default=30,
                        help="Number of runs to perform for each dataset. Default is 30.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    num_runs = args.num_runs
    seeds = [int.from_bytes(os.urandom(4), byteorder='little', signed=False) for _ in range(num_runs)]
    datasets = DATASETS_UCR_2018 if args.dataset_type == 'UCR' else DATASETS_UEA

    if args.dataset_name:
        datasets = [args.dataset_name]

    for dataset_name in datasets:
        dataset_metric_dict = defaultdict(list)
        for i, seed in enumerate(seeds):
            temp_conf = cfg.copy()
            temp_conf['seed'] = seed
            run_dict = evaluate(temp_conf, dataset_name)
            dataset_metric_dict['dataset'].append(dataset_name)

            for k, v in run_dict.items():
                dataset_metric_dict[k].append(float(v))
            if IS_HYPER_SEARCH:
                return run_dict['test_Accuracy']
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
