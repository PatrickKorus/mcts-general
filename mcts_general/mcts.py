import math
import sys

import gym
import numpy

from mcts_general.config import MCTSAgentConfig
from mcts_general.game import DeepCopyableGymGame
from common.wrapper import DeepCopyableWrapper


def select_action(node, temperature):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function
    in the config.
    """
    visit_counts = numpy.array(
        [child.visit_count for child in node.children.values()], dtype="int32"
    )
    actions = [action for action in node.children.keys()]
    if temperature == 0:
        action = actions[numpy.argmax(visit_counts)]
    elif temperature == float("inf"):
        action = numpy.random.choice(actions)
    else:
        # See paper appendix Data Generation
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )
        action = numpy.random.choice(actions, p=visit_count_distribution)
    return action


class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        observation,
        reward,
        done,
        game: DeepCopyableGymGame,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            # root_predicted_value = None
        else:
            root = Node(0)
            # root_predicted_value = 0    # Not used
            reward = reward
            policy_values = self.get_policy_priors()
            # hidden_state = observation
            #root_predicted_value = models.support_to_scalar(
            #    root_predicted_value, self.config.support_size
            #).item()
            # reward = models.support_to_scalar(reward, self.config.support_size).item()
            #assert (
            #    legal_actions
            #), f"Legal actions should not be an empty array. Got {legal_actions}."
            #assert set(legal_actions).issubset(
            #    set(self.config.action_space)
            #), "Legal actions should be a subset of the action space."
            root.expand(
                self.config.action_space,
                reward,
                observation=observation,
                game=game.get_copy(),
                done=done,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            if not parent.done:
                game_copy = parent.game.get_copy()

                observation, reward, done = game_copy.step(action)
                # TODO: Value for CartPole-v0
                value = reward if not done else 0
                # TODO: Value estimate over roll outs
                # value, reward, policy_logits, hidden_state = model.recurrent_inference(
                #     parent.hidden_state,
                #     torch.tensor([[action]]).to(parent.hidden_state.device),
                # )
                # value = models.support_to_scalar(value, self.config.support_size).item()
                # reward = models.support_to_scalar(reward, self.config.support_size).item()
                node.expand(
                    self.config.action_space,
                    reward,
                    observation,
                    game_copy,
                    done
                )
            else:
                value = 0

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            # "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def get_policy_priors(self):
        policy_values = {}  # TODO: What was that again?
        for action in self.config.action_space:
            policy_values[action] = 1  # TODO
        return policy_values

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        # prior_score = pb_c * child.prior
        prior_score = pb_c * (1 / len(parent.children)) # uniform prior score

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * child.value()
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        # if len(self.config.players) == 1:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())

            value = node.reward + self.config.discount * value

        # elif len(self.config.players) == 2:
        #     for node in reversed(search_path):
        #         node.value_sum += value # if node.to_play == to_play else -value
        #         node.visit_count += 1
        #         min_max_stats.update(node.reward + self.config.discount * -node.value())
        #
        #         value = (
        #             -node.reward if node.to_play == to_play else node.reward
        #         ) + self.config.discount * value

        # else:
        #     raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior=1):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.observation = None
        self.reward = 0
        self.done = False
        self.game = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        elif self.done:
            return 0 # TODO
        return self.value_sum / self.visit_count

    def expand(self, actions, reward, observation, game, done):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.done = done
        self.reward = reward
        self.observation = observation
        self.game = game
        for action in actions:
            self.children[action] = Node()

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    config = MCTSAgentConfig()
    mcts = MCTS(config)
    #env = gym.make("CartPole-v1")
    env = gym.make("MountainCar-v0")
    #env = DiscreteActionWrapper(env)
    #env = ScaledRewardWrapper(env, min_rew=-16.2736044, max_rew=0)
    env = DeepCopyableWrapper(env)
    game = DeepCopyableGymGame(env)
    game_copy = None
    obs = game.reset()
    done = False
    reward = 0
    render = True
    it = 0
    next_node = None
    while not done:
        it += 1
        result_node, info = mcts.run(
            observation=obs,
            reward=reward,
            game=game,
            add_exploration_noise=False,
            override_root_with=next_node
        )
        print(info)
        print(["{}: {}, ".format(action, child.visit_count) for action, child in result_node.children.items()])
        action = select_action(
            node=result_node,
            temperature=0,
        )
        next_node = result_node.children[action]
        print(action)
        if render:
            if game_copy is not None:
                game_copy.env.close()
            game_copy = game.get_copy()
            game_copy.env.render()
        obs, reward, done = game.step(action)
        print(reward)
    print(it)
