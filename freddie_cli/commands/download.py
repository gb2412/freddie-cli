import subprocess
import sys
import click


@click.command()
@click.option('--refresh', '-r', is_flag=True, help='Force a refresh of the data.')
def download(refresh):
    """
    CLI command to download Freddie Mac data.
    """

    cmd = [
        sys.executable, 
        "-m",
        "download.main"
    ]
    if refresh:
        cmd.append("--refresh")
    
    subprocess.run(cmd, check=True)