from owg_robot.ui import RobotEnvUI
from owg.utils.config import load_config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_objects', help='Overload number of objects to load', type=int, default=None)
parser.add_argument('--seed', help='Overload random seed', type=int, default=None)
parser.add_argument('--vis', help='Overload visualizer plotting', type=int, default=None)
parser.add_argument('--verbose', help='Overload VLM output verbocity', type=int, default=None)
kwargs = vars(parser.parse_args())

cfg = load_config('./config/pyb/env.yaml')
cfg.n_objects = kwargs['n_objects'] or cfg.n_objects
cfg.seed = kwargs['seed'] or cfg.seed
cfg.policy.vis = kwargs['vis'] or cfg.policy.vis
cfg.policy.verbose = kwargs['verbose'] or cfg.policy.verbose
print(cfg)

demo = RobotEnvUI(cfg)

demo.run()