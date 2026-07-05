"""Setuptools shim; full pyproject.toml packaging is roadmap issue #6."""
from setuptools import setup, find_packages

setup(
    name="voight-kampff",
    version="0.1",
    packages=find_packages(),
)
