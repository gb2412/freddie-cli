import click
from freddie_cli.commands.download import download
from freddie_cli.commands.process import process
from freddie_cli.commands.sample import sample
from freddie_cli.commands.train import train

@click.group()
def cli():
    """Freddie Mac CLI tool"""
    pass

cli.add_command(download)
cli.add_command(process)
cli.add_command(sample)
cli.add_command(train)

def main():
    cli()

if __name__ == '__main__':
    main()