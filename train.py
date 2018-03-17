#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.gameplay.snake import Snake
from snakeai.utils.cli import HelpOnFailArgumentParser
from tensorforce.agents import PPOAgent, Agent
from tensorforce.execution import Runner


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--agent',
        required=True,
        type=str,
        help='JSON file containing a agent definition.',
    )
    parser.add_argument(
        '--network',
        required=True,
        type=str,
        help='JSON file containing a network_spec definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    model = Sequential()

    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first',
        input_shape=(num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)

    environment = Snake(environment=env)

    # Network as list of layers
    # - Embedding layer:
    #   - For Gym environments utilizing a discrete observation space, an
    #     "embedding" layer should be inserted at the head of the network spec.
    #     Such environments are usually identified by either:
    #     - class ...Env(discrete.DiscreteEnv):
    #     - self.observation_space = spaces.Discrete(...)


    if parsed_args.network:
        with open(parsed_args.network, 'r') as fp:
            network_spec = json.load(fp=fp)

    if parsed_args.agent is not None:
        with open(parsed_args.agent, 'r') as fp:
            agent = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec
        )
    )

    # agent = PPOAgent(
    #     states=environment.states,
    #     actions=environment.actions,
    #     network=network_spec,
    #     # Agent
    #     states_preprocessing=None,
    #     actions_exploration=None,
    #     reward_preprocessing=None,
    #     # MemoryModel
    #     update_mode=dict(
    #         unit='episodes',
    #         # 10 episodes per update
    #         batch_size=20,
    #         # Every 10 episodes
    #         frequency=20
    #     ),
    #     memory=dict(
    #         type='latest',
    #         include_next_states=False,
    #         capacity=5000
    #     ),
    #     # DistributionModel
    #     distributions=None,
    #     entropy_regularization=0.01,
    #     # PGModel
    #     baseline_mode='states',
    #     baseline=dict(
    #         type='mlp',
    #         sizes=[32, 32]
    #     ),
    #     baseline_optimizer=dict(
    #         type='multi_step',
    #         optimizer=dict(
    #             type='adam',
    #             learning_rate=1e-3
    #         ),
    #         num_steps=5
    #     ),
    #     gae_lambda=0.97,
    #     # PGLRModel
    #     likelihood_ratio_clipping=0.2,
    #     # PPOAgent
    #     step_optimizer=dict(
    #         type='adam',
    #         learning_rate=1e-3
    #     ),
    #     subsampling_fraction=0.2,
    #     optimization_steps=25
    # )

    # Create the runner
    runner = Runner(agent=agent, environment=environment)

    # Callback function printing episode statistics
    def episode_finished(r):
        print(
            "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1]))
        return True

    # Start learning
    runner.run(episodes=parsed_args.num_episodes, max_episode_timesteps=1000, episode_finished=episode_finished)
    runner.close()

    # Print statistics
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:]))
    )


if __name__ == '__main__':
    main()
