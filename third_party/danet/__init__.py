"""Minimal vendored subset of official WhatAShot/DANet (Huang/Chen et al., AAAI 2022).

Upstream: https://github.com/WhatAShot/DANet
Commit:   b007c57121ec9082f6ef19ec7465d9df70767c26

Only AbstractLayer / ABSTLAY building blocks and Entmax15 are vendored.
Training loop, QHAdam, yacs, and dataset loaders are not included.
"""

from .DANet import AbstractLayer, BasicBlock, DANet, GBN, LearnableLocality  # noqa: F401
from .sparsemax import Entmax15, Sparsemax  # noqa: F401

UPSTREAM_URL = "https://github.com/WhatAShot/DANet"
UPSTREAM_COMMIT = "b007c57121ec9082f6ef19ec7465d9df70767c26"
PAPER = "Chen et al., DANets: Deep Abstract Networks for Tabular Data Classification and Regression, AAAI 2022"

__all__ = [
    "AbstractLayer",
    "BasicBlock",
    "DANet",
    "GBN",
    "LearnableLocality",
    "Entmax15",
    "Sparsemax",
    "UPSTREAM_URL",
    "UPSTREAM_COMMIT",
    "PAPER",
]
