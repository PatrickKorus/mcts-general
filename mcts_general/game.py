import typing

import abc
from copy import deepcopy

import gym

from common.wrapper import DeepCopyableWrapper, DiscreteActionWrapper


class DeepCopyableGame(metaclass=abc.ABCMeta):

    def __init__(self, seed):
        self.seed = seed

    @abc.abstractmethod
    def legal_actions(self) -> typing.List[typing.Union[float, int]]:
        pass

    @abc.abstractmethod
    def sample_action(self):
        pass

    @abc.abstractmethod
    def step(self, action) -> tuple:
        pass

    def render(self, mode='human', **kwargs):
        pass

    @abc.abstractmethod
    def get_copy(self):
        pass


class DeepCopyableGymGame:

    def __init__(self, env: gym.Env, seed=0):
        # if not isinstance(env.action_space, gym.spaces.Discrete):
        #    raise ValueError("Gym Env must have discrete action space!")
        self.env = DeepCopyableWrapper(env)
        self.env.seed(seed)
        self.is_rendering = False

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    @property
    def legal_actions(self):
        return [i for i in range(self.env.action_space.n)]

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        observation, rew, done, _ = self.env.step(action)
        return observation, rew, done

    def render(self, mode='human', **kwargs):
        # This workaround is necessary because a game / a gym env that is rendering cannot be deepcopied
        if not self.is_rendering:
            self.render_copy = self.get_copy()
            self.render_copy.env.render(mode, **kwargs)
            self.is_rendering = True
        else:
            self.render_copy.close()
            self.render_copy = self.get_copy()
            self.render_copy.env.render(mode, **kwargs)

    def get_copy(self):
        return DeepCopyableGymGame(deepcopy(self.env))


class PendulumGame(DeepCopyableGymGame):

    def __init__(self, num_actions, action_damping, seed=0):
        env = gym.make("Pendulum-v0")
        env = DiscreteActionWrapper(env, num_actions=num_actions, damping=action_damping)
        super(PendulumGame, self).__init__(env=env, seed=seed)
