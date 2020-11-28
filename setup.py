import subprocess
import sys

requirements = [
    "torchaudio==0.4.0",
    "torch==1.4.0",
    "dahuffman",
    "numpy"
]


def install_requirements():
    for package in requirements:
        install(package)


def install(package_name):
    subprocess.run([sys.executable, "-m", "pip", "install", package_name])


install_requirements()
