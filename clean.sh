#!/bin/sh
find . -type  f -name '*checkpoint.py' -delete -o -type  f -name '*checkpoint.yaml' -delete
find . -type d -name '*checkpoints' -delete
find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
rm -r Output
rm *.txt