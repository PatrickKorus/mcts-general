from collections.abc import Iterable

import numpy
import typing

import abc
from copy import deepcopy

import gym

from common.wrapper import DeepCopyableWrapper, DiscreteActionWrapper


class DeepCopyableGame(metaclass=abc.ABCMeta):

    def __init__(self, seed):
        self.rand = numpy.random
        self.rand.seed(seed)

    @abc.abstractmethod
    def legal_actions(self, simulation=False) -> typing.List[typing.Union[float, int]]:
        """ Used in tree expansion. """
        pass

    def sample_action(self, simulation=False):
        """ Used in Roll outs. """
        legal_actions = self.legal_actions(simulation=simulation)
        return legal_actions[self.rand.random_integers(0, len(legal_actions))]

    @abc.abstractmethod
    def step(self,
             action,
             simulation=False) -> tuple:
        pass

    def render(self, mode='human', **kwargs):
        pass

    @abc.abstractmethod
    def get_copy(self) -> "DeepCopyableGame":
        pass


class DeepCopyableGymGame(DeepCopyableGame):

    def __init__(self, env: gym.Env, seed=0):
        assert isinstance(env.action_space, gym.spaces.Discrete), "Gym Env must have discrete action space!"
        self.env = DeepCopyableWrapper(env)
        self.env.seed(seed)
        self.render_copy = None
        super(DeepCopyableGymGame, self).__init__(seed)

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
        if self.render_copy is not None:
            self.render_copy.close()

    def legal_actions(self, simulation=False):
        return [i for i in range(self.env.action_space.n)]

    def step(self, action, simulation=False):
        obs, rew, done, _ = self.env.step(int(action))
        return obs, rew, done

    def render(self, mode='human', **kwargs):
        # This workaround is necessary because a game / a gym env that is rendering cannot be deepcopied
        if self.render_copy is None:
            self.render_copy = self.get_copy()
            self.render_copy.env.render(mode, **kwargs)
        else:
            self.render_copy.close()
            self.render_copy = self.get_copy()
            self.render_copy.env.render(mode, **kwargs)

    def get_copy(self) -> "DeepCopyableGymGame":
        return DeepCopyableGymGame(deepcopy(self.env), seed=self.rand.randint(1e9))


class GymGameWithMacroActions(DeepCopyableGymGame):
    def __init__(self, env, seed, macro_actions: typing.List[typing.List[float]]):
        self._macro_actions = macro_actions
        super(GymGameWithMacroActions, self).__init__(env, seed)

    @property
    def macro_actions(self):
        return self._macro_actions

    def legal_actions(self, simulation=False):
        if simulation:
            # in simulation get the indexes of macro actions
            return [i for i in range(len(self.macro_actions))]
        else:
            # in evaluation get the indexes of the environment's action
            return [i for i in range(self.env.action_space.n)]

    def step(self, action, simulation=False):

        if simulation:
            # in simulation, traverse through the complete macro action
            reward = 0.
            mac_act = self.macro_actions[action]
            for a in mac_act:
                obs, rew, done = super(GymGameWithMacroActions, self).step(a)
                reward += rew
            reward /= len(mac_act)
        else:
            # in evaluation just take one step
            obs, reward, done = super(GymGameWithMacroActions, self).step(action)

        return obs, reward, done

    def get_copy(self) -> "GymGameWithMacroActions":
        return GymGameWithMacroActions(
            deepcopy(self.env),
            seed=self.rand.randint(1e9),
            macro_actions=self.macro_actions
        )


class GymGameDoingMultipleStepsInSimulations(GymGameWithMacroActions):

    def __init__(self, env, seed=0, number_of_multiple_actions_in_simulation=1):
        self.n = number_of_multiple_actions_in_simulation
        # macro actions are multiple actions i.e. >>> [[0, 0, 0, ...], [1, 1, 1, 1, ...], ...]
        macro_actions = [numpy.ones(self.n) * action for action in self.legal_actions()]
        super(GymGameDoingMultipleStepsInSimulations, self).__init__(env, seed, macro_actions)

    def get_copy(self) -> "GymGameDoingMultipleStepsInSimulations":
        return GymGameDoingMultipleStepsInSimulations(
            deepcopy(self.env),
            seed=self.rand.randint(1e9),
            number_of_multiple_actions_in_simulation=self.n
        )


class PendulumGameWithEngineeredMacroActions(GymGameWithMacroActions):

    def __init__(self, num_actions, action_damping, seed=0, max_macro_action_len=50):
        env = gym.make("Pendulum-v0")
        self.n_act = num_actions
        self.damping = action_damping
        env = DiscreteActionWrapper(env, num_actions=num_actions, damping=action_damping)
        self._max_macro_action_len = max_macro_action_len
        # macro actions are generated in each step
        super(PendulumGameWithEngineeredMacroActions, self).__init__(env=env, seed=seed, macro_actions=[])

    @property
    def macro_actions(self):
        macro_actions = []
        for action in super(PendulumGameWithEngineeredMacroActions, self).legal_actions(simulation=False):
            game_copy = self.get_copy()
            [cos_theta, sin_theta, theta_dot], _, done = game_copy.step(action)
            sign = numpy.sign(theta_dot)
            it = 1
            while sign == numpy.sign(theta_dot) and it <= self._max_macro_action_len and not done:
                [cos_theta, sin_theta, theta_dot], _, done = game_copy.step(action)
                it += 1
            macro_actions.append(numpy.ones(it) * action)
        return macro_actions

    def get_copy(self) -> "PendulumGameWithEngineeredMacroActions":
        copy = PendulumGameWithEngineeredMacroActions(num_actions=self.n_act,
                                                      action_damping=self.damping,
                                                      seed=self.rand.randint(1e9),
                                                      max_macro_action_len=self._max_macro_action_len)
        copy.env = deepcopy(self.env)
        return copy
