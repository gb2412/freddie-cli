import subprocess
import sys
import click


@click.command()
@click.option('--refresh', '-r', is_flag=True, help='Force a refresh of the data.')
@click.option('--use-econ-data', '-eco', is_flag=True, help='Include economic data.')
@click.option('--binary', '-bin', is_flag=True, help='Include economic data.')
def sample(refresh, use_econ_data, binary):
    """
    CLI command to sample Freddie Mac and economic data.
    """

    cmd = [
        sys.executable, 
        "-m",
        "sample.main"
    ]
    if refresh:
        cmd.append("--refresh")
    if use_econ_data:
        cmd.append("--use-econ-data")
    if binary:
        cmd.append("--binary")
    
    subprocess.run(cmd, check=True)