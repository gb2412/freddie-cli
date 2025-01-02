import subprocess
import sys
import click


@click.command()
@click.option('--model', '-m', type=str, required=True,
              help='Model to train (must be defined in config.yml)')
def train(model):
    """
    CLI command to train ML models on Freddie Mac and economic data.
    """
    cmd = [
        sys.executable,
        "-m",
        "train.main",
        "--model", model
    ]
    subprocess.run(cmd, check=True)
