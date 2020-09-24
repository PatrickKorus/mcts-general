import math
import numpy

from mcts_general.common.normalize import MinMaxStats
from mcts_general.config import MCTSAgentConfig, MCTSContinuousAgentConfig
from mcts_general.game import DeepCopyableGame, ContinuousGymGame


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


def get_roll_out(game: DeepCopyableGame, n, max_depth=None, discount=0.995, do_simulation_steps=False):
    total_reward = 0
    done = True
    for _ in range(n):
        trajectory_reward = 0
        game_copy = game.get_copy()
        for it in range(max_depth):
            if done:
                break
            action = game_copy.sample_action()
            _, reward, done = game_copy.step(action, simulation=do_simulation_steps)
            trajectory_reward += discount * reward
        total_reward += trajectory_reward / (it + 1)
    return total_reward


class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config: MCTSAgentConfig):
        self.config = config
        self.node_cls = Node

    def run(
        self,
        observation,
        reward,
        done,
        game: DeepCopyableGame,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with is not None:
            root = override_root_with
        else:
            root = self.node_cls(0)
            reward = reward
            root.expand(
                observation,
                reward,
                done,
                game.get_copy(),
                initial_visit_count=0
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
                observation, reward, done = game_copy.step(action, simulation=True)

                if self.config.do_roll_outs:
                    value = get_roll_out(game_copy,
                                         self.config.number_of_roll_outs,
                                         self.config.max_roll_out_depth,
                                         self.config.discount,
                                         do_simulation_steps=self.config.do_roll_out_steps_with_simulation_true)
                    initial_visit_count = self.config.number_of_roll_outs - 1    # -1 because of increment in backprop
                else:
                    value = reward
                    initial_visit_count = 0

                node.expand(
                    observation,
                    reward,
                    done,
                    game_copy,
                    initial_visit_count=initial_visit_count
                )
            else:
                value = 0

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
        }
        return root, extra_info

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
        prior_score = pb_c * (1 / len(parent.children))     # uniform prior score

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
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())

            value = node.reward + self.config.discount * value


class Node:

    def __init__(self, prior=1):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior      # uniform prior
        self.value_sum = 0
        self.children = {}
        self.observation = None
        self.reward = 0
        self.done = False
        self.game = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0 or self.done:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, observation, reward, done, game, initial_visit_count=0):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network. TODO Doc
        """
        self.done = done
        self.reward = reward
        self.observation = observation
        self.game = game
        self.visit_count = initial_visit_count
        for action in game.legal_actions(simulation=True):
            self.children[action] = Node()

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions. TODO Doc
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class ContinuousNode(Node):

    def expand(self, observation, reward, done, game, initial_visit_count=0):
        self.reward = reward
        self.observation = observation
        self.game = game
        self.visit_count = initial_visit_count
        action = game.sample_action(simulation=True)
        self.children[action] = ContinuousNode()


class ContinuousMCTS(MCTS):
    def __init__(self, config: MCTSContinuousAgentConfig):
        super(ContinuousMCTS, self).__init__(config)
        self.config = config
        self.node_cls = ContinuousNode

    def run(
            self,
            observation,
            reward,
            done,
            game: ContinuousGymGame,
            add_exploration_noise,
            override_root_with=None,
    ):
        return super(ContinuousMCTS, self).run(
            observation,
            reward,
            done,
            game,
            add_exploration_noise,
            override_root_with
        )

    def select_child(self, node, min_max_stats):
        # Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
        C = self.config.C
        alpha = self.config.alpha
        while len(node.children) < math.ceil(C * node.visit_count ** alpha):
            action = node.game.sample_action(simulation=True)
            node.children[action] = ContinuousNode()

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
