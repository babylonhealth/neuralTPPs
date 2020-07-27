import argparse
from ruamel.yaml import YAML


def main(args):
    yaml = YAML()
    with open(args.input_config_path, "r") as stream:
        config = yaml.load(stream)

    config['spec']['containers'][0]['image'] = args.image

    with open(args.output_config_path, "w") as stream:
        yaml.dump(config, stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config_update")
    parser.add_argument("--input-config-path", type=str,
                        default="deploy/interactive.yml",
                        help="The deploy file to update.")
    parser.add_argument("--output-config-path", type=str,
                        default="deploy/interactive-auto.yml",
                        help="The save location of the updated deploy file.")
    parser.add_argument("--image", type=str, default=None,
                        help="The image to replace in the deploy file.")
    parser.add_argument("--allow-uncommitted", action='store_true',
                        help="allow uncommitted changes for "
                             "runs (default=False)")
    parser.set_defaults(allow_uncommitted=False)
    parsed_args = parser.parse_args()
    main(args=parsed_args)
