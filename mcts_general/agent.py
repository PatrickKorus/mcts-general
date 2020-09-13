from mcts_general.config import MCTSAgentConfig
from mcts_general.game import DeepCopyableGymGame
from mcts_general.mcts import MCTS, select_action


class MCTSAgent:

    def __init__(self, config: MCTSAgentConfig):
        self.config = config
        self.mcts = MCTS(self.config)
        self.result_node = None

    def step(self, game_state: DeepCopyableGymGame, observations, reward):
        self.result_node, info = self.mcts.run(
            observation=observations,
            reward=reward,
            game=game_state,
            add_exploration_noise=False,
            override_root_with=self.result_node if self.config.reuse_tree else None
        )
        return select_action(self.result_node, temperature=self.config.temperature)
