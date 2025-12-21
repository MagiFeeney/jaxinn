from typing import Optional, Tuple, Literal
import tyro

from agent import Agent
from dataclasses import dataclass


@dataclass
class Args:
    # General
    agent_name: Literal['PSRL', 'EUBRL', 'QLearning', 'RMAX', 'SARSA', 'VBRB', 'BEB'] = 'EUBRL'
    """the agent you wish to choose"""


def main():
    pass


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
