import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="path to model")

args = parser.parse_args()
