#! /usr/bin/env python3
"""
"""
from setuptools import setup, find_packages

VERSION = "0.0.0"
DESCRIPTION = "A Python package for exercising convolution"

setup(
    name="convolution",
    version=VERSION,
    description=DESCRIPTION,
    author="Hossam Khair",
    author_email="hossam.khir1@outlook.com",
    packages=find_packages(),
    install_requires=["numpy"],
    keywords=["convolution"],
)
