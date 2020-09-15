class MCTSAgentConfig:

    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.999
        self.action_space = [i for i in range(2)]
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.num_simulations = 400
        self.reuse_tree = False
        self.temperature = 0
