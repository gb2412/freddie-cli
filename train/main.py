import sys
import importlib

from utils.config import get_model_config


def load_model_function(model_name):
    """Dynamically import model training function"""

    model_configs = get_model_config(model_name)
    module = importlib.import_module('train.'+model_name)
    train_fn = getattr(module, 'train_'+model_name)
    return train_fn, model_configs


def parse_model_arg():
    """Parse and validate model argument"""
    # Get all argument except script name
    args = sys.argv[1:]

    # Check if --model flag exists
    if not args or '--model' not in args:
        raise ValueError("Specify model to train using: freddie train --model <model_name>")
    # Get value for --model flag
    try:
        model_index = args.index('--model')
        return args[model_index + 1]
    except IndexError:
        raise ValueError("No model name provided after --model flag")


def main():
    """Train specified model with config parameters"""

    # Get model name
    model_name = parse_model_arg()
    # Load model configurations and training function
    train_fn, model_configs = load_model_function(model_name)

    # Call training function with parameters
    train_fn(**model_configs)


if __name__ == "__main__":
    main()