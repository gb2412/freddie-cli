from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="freddie-cli",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=read_requirements('requirements.txt'),
    entry_points={
        "console_scripts": [
            "freddie=freddie_cli.cli:main"
        ]
    }
)