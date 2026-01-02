# Instantiate the parser
import argparse


parser = argparse.ArgumentParser()

# Optional argument
parser.add_argument('--transport', type=str, help='MCP Transport type', default='stdio')

args = parser.parse_args()