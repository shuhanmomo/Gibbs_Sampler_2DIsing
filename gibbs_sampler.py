from graph_helpers import Ising2D, TreeSampler
import numpy as np
import random
import matplotlib.pyplot as plt


class GibbsSampler:
    """Gibbs sampler for the Ising model."""

    def __init__(self, ising_model: Ising2D, initial_state="random"):
        """
        Initialize the Gibbs sampler with an Ising model and an initial state.

        input:
            ising_model (Ising2D): The Ising model instance to sample from.
            initial_state (str or dict): 'random', 'all_pos', 'all_neg'
        """
        self.ising_model = ising_model
        self.N = ising_model.N
        self.theta = ising_model.theta
        self.X = ising_model.X  # valuse x can take e.g., [-1, 1]
        self.nodes = ising_model.Ising.get_V()
        self.edges = ising_model.Ising.get_E()
        self.current_state = self.initialize_state(initial_state)

    def plot_samples(self, collected_samples, iterations):
        """
        Plot the collected samples in a row with proper notation and titles.

        input:
            collected_samples (list): List of sample states (dicts).
            iterations (list): List of iteration numbers corresponding to the samples.
        """
        num_samples = len(collected_samples)
        if num_samples == 0:
            print("No samples collected to plot.")
            return

        # Determine the number of columns and rows
        cols = min(5, num_samples)  # Adjust as needed
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()  # Flatten the axes array

        for idx, (sample_state, ax, iter_num) in enumerate(
            zip(collected_samples, axes, iterations)
        ):
            self.ising_model.visualize(sample_state, ax=ax, iteration=iter_num)

        # Hide any unused subplots
        for ax in axes[num_samples:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def initialize_state(self, initial_state):
        """
        Initialize the state of the spins.

        input:
            initial_state (str): 'random', 'all_pos', 'all_neg'
        return:
            dict: A dictionary mapping each node index to its spin value.
        """
        state = {}
        if initial_state == "random":
            for node in self.nodes:
                state[node] = np.random.choice(self.X)
        elif initial_state == "all_pos":
            for node in self.nodes:
                state[node] = self.X[0]
        elif initial_state == "all_neg":
            for node in self.nodes:
                state[node] = self.X[1]
        else:
            raise ValueError(
                "Invalid initial_state. Must be 'random', 'all_pos', 'all_neg'"
            )
        return state

    def node_by_node_gibbs(
        self, num_iterations=1000, plot_interval=100, initial_state="random"
    ):
        """
        Perform node-by-node Gibbs sampling.

        input:
            num_iterations (int): Number of iterations to run the sampler.
            plot_every (int): Plot the grid every 'plot_every' iterations.
        """
        self.current_state = self.initialize_state(initial_state)
        collected_samples = []
        iterations = []
        for iteration in range(1, num_iterations + 1):
            for node in self.nodes:
                # Compute the local field h_k
                neighbors = self.ising_model.Ising.get_neighbors(node)
                sum_neighbor_spins = sum(
                    self.current_state[neighbor] for neighbor in neighbors
                )
                h_k = self.theta * sum_neighbor_spins

                # Compute the probability p_k of X_k = +1
                p_k = 1 / (1 + np.exp(-2 * h_k))

                # Sample X_k from Bernoulli(p_k)
                self.current_state[node] = np.random.choice(self.X, p=[1 - p_k, p_k])
            if iteration % plot_interval == 0:
                print(f"Iteration {iteration}")
                collected_samples.append(self.current_state.copy())
                iterations.append(iteration)
        self.plot_samples(collected_samples, iterations)

    def block_gibbs_sampler(
        self, num_iterations=1000, plot_interval=100, initial_state="random"
    ):
        """
        Perform block Gibbs sampling by alternating between two subgraphs (trees).

        input:
            num_iterations (int): Number of iterations to run the sampler.
            plot_every (int): Plot the grid every 'plot_every' iterations.
        """
        # Split the graph into two trees
        self.current_state = self.initialize_state(initial_state)
        tree_A, tree_B = self.ising_model.split_and_calculate_trees()
        sampler_A = TreeSampler(tree_A, X=self.X)
        sampler_B = TreeSampler(tree_B, X=self.X)

        # Initialize messages
        sampler_A.root_idx = 0
        sampler_B.root_idx = 0

        collected_samples = []
        iterations = []

        for iteration in range(1, num_iterations + 1):
            # Sample P(A | B)
            sampler_A.messages = None  # Reset messages
            sampler_A.sum_product()
            sample_A = sampler_A.sample()

            # Update A nodes
            for node in tree_A.get_V():
                self.current_state[node] = sample_A[node]

            # Sample P(B | A)
            sampler_B.messages = None  # Reset messages
            sampler_B.sum_product()
            sample_B = sampler_B.sample()

            # Update B nodes
            for node in tree_B.get_V():
                self.current_state[node] = sample_B[node]

            # Plot
            if iteration % plot_interval == 0:
                print(f"Iteration {iteration}")
                collected_samples.append(self.current_state.copy())
                iterations.append(iteration)
        self.plot_samples(collected_samples, iterations)


if __name__ == "__main__":
    N = 60  # grid size
    theta = 0.45  # param
    X = [-1, 1]  # x values
    ising_model = Ising2D(N, theta, X)
    gibbs_sampler = GibbsSampler(ising_model, initial_state="random")
    # node-by-node sampler
    gibbs_sampler.node_by_node_gibbs(num_iterations=1000, plot_interval=100)
    # block gibbs sampler
    # gibbs_sampler.block_gibbs_sampler(num_iterations=1000, plot_interval=100)
