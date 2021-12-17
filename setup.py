# import setuptools
import setuptools
from setuptools import setup

setup(
    name='sql',
    version='1.0',
    description='sequential reinforcement learning with transformers',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'transformers',
        'gym',
        'matplotlib',
        'mujoco-py<2.2,>=2.1'
    ]
)