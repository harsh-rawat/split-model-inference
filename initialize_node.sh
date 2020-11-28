#!/bin/bash

sudo update-alternatives --remove python /usr/bin/python2

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10

sudo apt update

sudo apt install python3-pip

python setup.py