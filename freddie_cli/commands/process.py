import subprocess
import sys
import click


@click.command()
@click.option('--refresh', '-r', is_flag=True, help='Force a refresh of the data.')
def process(refresh):
    """
    CLI command to process Freddie Mac data.
    """

    cmd = [
        sys.executable, 
        "-m",
        "process.main"
    ]
    if refresh:
        cmd.append("--refresh")
    
    subprocess.run(cmd, check=True)