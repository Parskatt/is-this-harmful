import argparse
import subprocess

parser = argparse.ArgumentParser(description='Submit a list of batch jobs')
parser.add_argument('configs', nargs='+', default=[])
args = parser.parse_args()
configs = args.configs

for config in configs:
    ret = subprocess.call(['python','tools/train.py',config])