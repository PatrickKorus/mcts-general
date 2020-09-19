from copy import deepcopy

import gym
from gym.spaces import Discrete
import numpy


class DiscreteActionWrapper(gym.ActionWrapper):

    def __init__(self, env, num_actions=2, damping=0.75):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.action_space = Discrete(num_actions)
        high = self.original_action_space.high
        low = self.original_action_space.low
        rnge = high - low
        mid = low + 0.5 * rnge
        high = mid + 0.5 * damping * rnge
        low = mid - 0.5 * damping * rnge
        step = (rnge * damping) / (num_actions - 1)
        self.action_equivalents = [numpy.array([i], dtype="float32") for i in numpy.arange(start=low,
                                               stop=high + 1,    # so that high is included
                                               step=step)[0:num_actions]]

    def action(self, act):
        return self.action_equivalents[act]

    def reverse_action(self, action):
        return self.action_equivalents.index(action)


class ScaledRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, min_rew, max_rew):
        super().__init__(env)
        self.min_rew = min_rew
        self.max_rew = max_rew

    def reward(self, reward):
        reward = ((reward - self.min_rew) / (self.max_rew - self.min_rew))  # * 2 - 1
        return reward


class DeepCopyableWrapper(gym.Wrapper):

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result


class ManipulatedTimeDiscretization(gym.Wrapper):

    def __init__(self, env, number_of_time_steps_to_scip):
        assert number_of_time_steps_to_scip > 0, 'Number of time steps to stack must be larger than 0!'
        self.n_scip = number_of_time_steps_to_scip
        super(ManipulatedTimeDiscretization, self).__init__(env)

    def step(self, action):
        for _ in range(self.n_scip):
            obs, rew, done, inf = self.env.step(action)
        return obs, rew, done, inf
