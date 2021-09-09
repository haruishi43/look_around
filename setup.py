#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="LookAround",
    version="0.0.1",
    packages=find_packages(
        exclude=["tests", "scripts", "results", "data", "dataset", "configs", "logs", "notebooks",]
    ),
    python_requires=">=3.6",
)
