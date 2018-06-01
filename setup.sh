#!/bin/bash

# Required because of using numpy-indexed, which is badly packaged:
pip2 install pyyaml --user

pip2 install -r requirements.txt --user

# For displaying in 3d
pip3 install vpython --user
