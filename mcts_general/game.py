from copy import deepcopy

import gym

from common.wrapper import DeepCopyableWrapper, DiscreteActionWrapper


class DeepCopyableGymGame:

    def __init__(self, env: gym.Env, seed=0):
        # if not isinstance(env.action_space, gym.spaces.Discrete):
        #    raise ValueError("Gym Env must have discrete action space!")
        self.env = DeepCopyableWrapper(env)
        self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    @property
    def legal_actions(self):
        # TODO: decide whether this should be in config or game
        return [i for i in range(self.env.action_space.n)]

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        observation, rew, done, _ = self.env.step([action])
        return observation, rew, done

    def get_copy(self):
        return DeepCopyableGymGame(deepcopy(self.env))


class PendulumGame(DeepCopyableGymGame):

    def __init__(self, num_actions, action_damping, seed=0):
        env = gym.make("Pendulum-v0")
        env = DiscreteActionWrapper(env, num_actions=num_actions, damping=action_damping)
        super(PendulumGame, self).__init__(env=env, seed=seed)
