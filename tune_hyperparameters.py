import argparse
import itertools
import json

from main import main

parser = argparse.ArgumentParser()

parser.add_argument('--hyperparameters_config', type=str, default='hyperparameters_config.json')

args = vars(parser.parse_args())


def tune_hyperparameters(config_path):
    with open(config_path) as f:
        config = json.loads(f.read())

    flag_names = list(config.keys())
    all_flag_values = list(config.values())
    flag_value_combinations = list(itertools.product(*all_flag_values))

    for flag_values in flag_value_combinations:
        args = dict(zip(flag_names, flag_values))
        main(args)


if __name__ == '__main__':
    tune_hyperparameters(args['hyperparameters_config'])
