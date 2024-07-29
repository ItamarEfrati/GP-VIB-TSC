import os

import hydra
import pyrootutils
import pandas as pd

from omegaconf import DictConfig

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/hmnist.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=os.path.join(root, "configs"), config_name="hmnist.yaml")
def main(cfg: DictConfig):
    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.tasks.eval_task import evaluate

    # evaluate the model
    metric_dict = {}
    num_runs = cfg.num_runs
    seeds = [int.from_bytes(os.urandom(4), byteorder='little', signed=False) for _ in range(num_runs)]
    for i, seed in enumerate(seeds):
        temp_conf = cfg.copy()
        temp_conf['seed'] = seed
        metric_dict[i] = evaluate(temp_conf)

    new_dict = {'run': list(range(1, num_runs + 1))}
    for k in list(metric_dict[0].keys()):
        new_dict[k] = list([float(v[k]) for v in metric_dict.values()])

    df = pd.DataFrame.from_dict(new_dict).set_index('run')
    mean = df.drop(columns='seed').mean()
    mean['seed'] = '-'
    mean.name = 'mean'
    std = df.drop(columns='seed').std()
    std['seed'] = '-'
    std.name = 'std'
    df = pd.concat([df, mean.to_frame().T], axis=0)
    df = pd.concat([df, std.to_frame().T], axis=0)
    df.to_csv(os.path.join(cfg.paths.output_dir, 'results.csv'))


if __name__ == "__main__":
    main()
