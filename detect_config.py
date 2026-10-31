import argparse
import json

from utils.runtime import detect_config, load_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path')
    args = parser.parse_args()
    print(json.dumps(detect_config(load_state(args.checkpoint_path)), indent=2))


if __name__ == '__main__':
    main()
