import pprint
import random
import time

import numpy as np
import pandas as pd

import tensorforce.environments.environment as env
from .entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS


class Snake(env.Environment):
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def close(self):
        pass

    def reset(self):
        result = self.environment.new_episode()
        return result.observation

    def execute(self, actions):
        self.environment.choose_action(actions)
        result = self.environment.timestep()
        return result.observation, result.is_episode_end, result.reward

    @property
    def states(self):
        return dict(
            shape=(self.environment.field.size, self.environment.field.size),
            type='float32')

    @property
    def actions(self):
        return dict(type='int', num_actions=self.environment.num_actions)

    @staticmethod
    def from_spec(spec, kwargs):
        return super().from_spec(spec, kwargs)

    def __init__(self, environment):
        self.environment = environment
