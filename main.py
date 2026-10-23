import json
import argparse

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    from trainer import train

    train(args)


def load_json(settings_path):
    with open(settings_path, encoding='utf-8') as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Train M2LE for continual learning.')
    parser.add_argument('--config', type=str, default='./exps/imgr10.json',
                        help='Path to the JSON experiment configuration.')
    parser.add_argument('--device', type=str, default='0')

    return parser


if __name__ == '__main__':
    main()
