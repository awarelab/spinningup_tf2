"""SpinningUp setup."""

import sys
import setuptools

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    'The Spinning Up repo is designed to work with Python 3.6 and greater. ' \
    'Please install it before proceeding.'

setuptools.setup(
    name='spinup_bis',
    py_modules=['spinup_bis'],
    version='0.1',
    install_requires=[
        'gym[atari,box2d,classic_control]~=0.15.3',
        'jupyter',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'tensorflow>=2.0',
    ],
    description='Teaching tools for introducing people to deep RL in TF2.',
)
