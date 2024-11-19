import numpy as np
from scipy.special import logsumexp

import matplotlib.pyplot as plt


class GraphModel:
    """
    Simple class for undirected graphical models over binary variables.
    """

    ## Initialized by passing in:
    ##    i) dictionary mapping from nodes to node potentials (1x2 np arrays)
    ##    ii) dictionary mapping from edges (i,j) pairs to edge potentials (2x2 np arrays)
    def __init__(self, node_potentials, edge_potentials):
        """
        eg: self.node_potentials[i][0] = 1,self.node_potentials[i][1] = exp(alphas[i-1])
        eg: self.edge_potentials[(i,j)][0][0] = exp(beta),self.edge_potentials[(i,j)][1][0] = 1
        self.edge_potentials[(i,j)][0][1] = 1, self.edge_potentials[(i,j)][1][1] = exp(beta)
        """
        self.node_potentials = node_potentials.copy()
        self.edge_potentials = edge_potentials.copy()
        self.nodes = self.get_V()
        self.edges = self.get_E()

    def get_V(self):
        return list(self.node_potentials.keys()).copy()

    def get_E(self):
        return list(self.edge_potentials.keys()).copy()

    def get_neighbors(self, i):
        """
        return the indexes of node i' neighbors
        """
        assert i in self.nodes
        neighbors = []
        for edge in self.edges:
            node1, node2 = edge
            if node1 == i and node2 not in neighbors:
                neighbors.append(node2)
            elif node2 == i and node1 not in neighbors:
                neighbors.append(node1)

        return neighbors


class Ising2D:
    """generate a simple 2D Ising Graph according to P_xv = 1/Z exp{sum{theta*x_i*x_j}}"""

    def __init__(self, N, theta, X=[-1, 1]):
        """
        N: size of ising graph (N*N nodes)
        theta: coupling parameter
        X: an array of possible x values
        """
        self.N = N
        self.theta = theta
        self.X = X
        self.Ising = self.generate_2Dising(N, theta, X)

    def generate_2Dising(self, N, theta, X):
        """
        Generate a 2D Ising model graph using the GraphModel class.
        """
        node_potentials = {}
        edge_potentials = {}

        # Define node potentials (assume uniform distribution over X for simplicity)
        for i in range(N):
            for j in range(N):
                node_index = (i, j)  # Node represented by its 2D coordinates
                node_potentials[node_index] = np.array(
                    [1, 1]
                )  # Uniform potential for each node

        # Define edge potentials based on the Ising model
        for i in range(N):
            for j in range(N):
                node_index = (i, j)

                # Add edges to the right and below (avoiding duplicates)
                if j + 1 < N:  # Horizontal neighbor
                    neighbor_index = (i, j + 1)
                    x0 = self.X[0]
                    x1 = self.X[1]
                    edge_potentials[(node_index, neighbor_index)] = np.array(
                        [
                            [np.exp(theta * x0 * x0), np.exp(theta * x0 * x1)],
                            [np.exp(theta * x1 * x0), np.exp(theta * x1 * x1)],
                        ]
                    )

                if i + 1 < N:  # Vertical neighbor
                    neighbor_index = (i + 1, j)
                    x0 = self.X[0]
                    x1 = self.X[1]
                    edge_potentials[(node_index, neighbor_index)] = np.array(
                        [
                            [np.exp(theta * x0 * x0), np.exp(theta * x0 * x1)],
                            [np.exp(theta * x1 * x0), np.exp(theta * x1 * x1)],
                        ]
                    )

        # Create a GraphModel instance with node and edge potentials
        return GraphModel(node_potentials, edge_potentials)

    def visualize(self, sample_state, ax=None, iteration=None):
        """
        Visualize the Ising model grid based on the sample_state.

        input:
            sample_state (dict): A dictionary mapping node indices to their spin values.
            ax (matplotlib.axes.Axes): The axis to plot on. If None, creates a new figure.
            iteration (int): Iteration number for the title.
        """
        # Initialize a 2D numpy array to hold the spin values
        grid = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                node_index = (i, j)
                spin_value = sample_state.get(
                    node_index, self.X[0]
                )  # Default to X[0] if not specified
                # Map the spin value to 0 or 1 for visualization
                grid[i, j] = self.X.index(spin_value)

        # Create a colormap: 0 -> black, 1 -> white
        cmap = plt.cm.gray
        if ax is None:
            plt.figure(figsize=(6, 6))
            ax = plt.gca()
        im = ax.imshow(grid, cmap=cmap, interpolation="nearest", origin="upper")
        ax.axis("off")
        if iteration is not None:
            ax.set_title(f"Iteration {iteration}")
        else:
            ax.set_title("Ising Model Visualization")

    def split_and_calculate_trees(self):
        """
        Split the Ising graph into disjoint sets A and B, and calculate the conditional potentials
        for P(A | B) and P(B | A).
        Returns:
            tree_A: GraphModel object representing P(A | B)
            tree_B: GraphModel object representing P(B | A)
        """
        # Step 1: Initialize sets and edge lists
        nodes_A = set()
        nodes_B = set()
        edges_AA = []
        edges_BB = []
        edges_AB = []

        # Step 1.1: Split nodes into A and B
        for i in range(self.N):
            for j in range(self.N):
                node_index = (i, j)
                if i == 0 or (
                    j % 2 == 0 and i < self.N - 1
                ):  # First row or odd column -> Tree A
                    nodes_A.add(node_index)
                if i == self.N - 1 or (
                    j % 2 == 1 and i > 0
                ):  # Last row or even column -> Tree B
                    nodes_B.add(node_index)

        # Step 1.2: Split edges into AA, BB, and AB
        for i in range(self.N):
            for j in range(self.N):
                node_index = (i, j)

                # Horizontal edges
                if j + 1 < self.N:  # Right neighbor
                    neighbor_index = (i, j + 1)
                    if node_index in nodes_A and neighbor_index in nodes_A:
                        edges_AA.append((node_index, neighbor_index))
                    elif node_index in nodes_B and neighbor_index in nodes_B:
                        edges_BB.append((node_index, neighbor_index))
                    else:
                        edges_AB.append((node_index, neighbor_index))

                # Vertical edges
                if i + 1 < self.N:  # Below neighbor
                    neighbor_index = (i + 1, j)
                    if node_index in nodes_A and neighbor_index in nodes_A:
                        edges_AA.append((node_index, neighbor_index))
                    elif node_index in nodes_B and neighbor_index in nodes_B:
                        edges_BB.append((node_index, neighbor_index))
                    else:
                        edges_AB.append((node_index, neighbor_index))

        # Calculate P(A | B)
        tree_A_node_potentials, tree_A_edge_potentials = self.calculate_tree_potentials(
            nodes_A, nodes_B, edges_AA, self.theta
        )

        # Calculate P(B | A)
        tree_B_node_potentials, tree_B_edge_potentials = self.calculate_tree_potentials(
            nodes_B, nodes_A, edges_BB, self.theta
        )

        # Create GraphModel objects
        tree_A = GraphModel(tree_A_node_potentials, tree_A_edge_potentials)
        tree_B = GraphModel(tree_B_node_potentials, tree_B_edge_potentials)

        return tree_A, tree_B

    # Step 2: Calculate conditional node and edge potentials
    def calculate_tree_potentials(self, tree_nodes, other_tree_nodes, edges, theta):
        node_potentials = {}
        edge_potentials = {}

        # Calculate node potentials
        for node in tree_nodes:
            neighbors = self.Ising.get_neighbors(node)
            potential = []
            for x_i in self.X:
                phi = 0
                for neighbor in neighbors:
                    if neighbor in other_tree_nodes:
                        for x_j in self.X:
                            phi += theta * x_i * x_j
                potential.append(np.exp(phi))
            node_potentials[node] = np.array(potential)

        # Calculate edge potentials
        for edge in edges:
            edge_potentials[edge] = self.Ising.edge_potentials[edge]

        return node_potentials, edge_potentials


class TreeSampler:
    """Implement tree sampler to sample from joint distribution of a tree."""

    def __init__(self, tree: GraphModel, X=[-1, 1]):
        self.tree_graph = tree
        self.root_idx = -1  # Should be set before sampling
        self.messages = None
        self.X = X  # Binary values x could take

    def compute_message(self, i, j, messages):
        """
        Computes the log-space message from node i to node j.
        """
        tree_graph = self.tree_graph

        # Check if the message (i, j) is already computed
        if (i, j) in messages:
            return messages[(i, j)]

        # Node potential for node i
        phi_i = tree_graph.node_potentials[i]
        log_phi_i = np.log(phi_i)

        # Get the neighbors of node i excluding j
        neighbors_i = [
            neighbor for neighbor in tree_graph.get_neighbors(i) if neighbor != j
        ]

        # Edge potential between i and j
        e_ij = (i, j) if (i, j) in tree_graph.edge_potentials else (j, i)
        psi_ij = tree_graph.edge_potentials[e_ij]
        log_psi_ij = np.log(psi_ij)

        if not neighbors_i:
            # Leaf node
            if e_ij == (i, j):
                # Compute log-message
                log_message_ij = np.zeros(len(self.X))
                for x_j in range(len(self.X)):
                    log_message_ij[x_j] = logsumexp(log_phi_i + log_psi_ij[:, x_j])
            else:
                log_message_ij = np.zeros(len(self.X))
                for x_j in range(len(self.X)):
                    log_message_ij[x_j] = logsumexp(log_phi_i + log_psi_ij.T[:, x_j])
        else:
            # Non-leaf node
            # Compute the sum of log-messages from neighbors
            log_product_messages = np.zeros(len(self.X))
            for neighbor in neighbors_i:
                log_message_ki = self.compute_message(neighbor, i, messages)
                log_product_messages += log_message_ki

            log_phi_i_message = log_phi_i + log_product_messages

            if e_ij == (i, j):
                log_message_ij = np.zeros(len(self.X))
                for x_j in range(len(self.X)):
                    log_sums = []
                    for x_i in range(len(self.X)):
                        log_sums.append(log_phi_i_message[x_i] + log_psi_ij[x_i, x_j])
                    log_message_ij[x_j] = logsumexp(log_sums)
            else:
                log_message_ij = np.zeros(len(self.X))
                for x_j in range(len(self.X)):
                    log_sums = []
                    for x_i in range(len(self.X)):
                        log_sums.append(log_phi_i_message[x_i] + log_psi_ij.T[x_i, x_j])
                    log_message_ij[x_j] = logsumexp(log_sums)

        # Normalize log-message to prevent numerical issues
        max_log_msg = np.max(log_message_ij)
        log_message_ij -= max_log_msg

        # Store the computed message (i, j)
        messages[(i, j)] = log_message_ij

        return log_message_ij

    def sum_product(self):
        tree_graph = self.tree_graph
        self.messages = {}
        nodes = tree_graph.get_V()
        root = nodes[self.root_idx]

        # Bottom-up pass: compute messages from leaves to root
        for neighbor in tree_graph.get_neighbors(root):
            self.compute_message(neighbor, root, self.messages)

        # Top-down pass: not necessary for sampling as we use the messages computed

    def sample(self):
        """
        Perform joint sampling of node values from the tree's joint distribution.

        Returns:
            dict: A dictionary mapping each node index to its sampled state
        """
        tree_graph = self.tree_graph
        if self.messages is None:
            self.sum_product()

        sampled_values = {}
        root = tree_graph.get_V()[self.root_idx]

        # Root marginal in log-space
        phi_root = tree_graph.node_potentials[root]
        log_phi_root = np.log(phi_root)
        log_marginal_root = log_phi_root.copy()
        for neighbor in tree_graph.get_neighbors(root):
            log_message = self.messages[(neighbor, root)]
            log_marginal_root += log_message

        # Convert log-marginal to probabilities
        log_marginal_root -= logsumexp(log_marginal_root)
        marginal_root = np.exp(log_marginal_root)

        # Sample root
        x_root = np.random.choice(self.X, p=marginal_root)
        sampled_values[root] = x_root

        def sample_children(parent, node):
            # Compute conditional distribution p(x_node | x_parent) in log-space
            phi_node = self.tree_graph.node_potentials[node]
            log_phi_node = np.log(phi_node)

            # Messages from other neighbors
            log_product_messages = np.zeros(len(self.X))
            for neighbor in self.tree_graph.get_neighbors(node):
                if neighbor != parent:
                    log_product_messages += self.messages[(neighbor, node)]

            # Fetch edge potential between parent and node and handle orientation
            if (parent, node) in self.tree_graph.edge_potentials:
                psi = self.tree_graph.edge_potentials[(parent, node)]
            elif (node, parent) in self.tree_graph.edge_potentials:
                psi = self.tree_graph.edge_potentials[(node, parent)].T
            else:
                raise ValueError(
                    f"Edge potential not found between nodes {parent} and {node}"
                )
            log_psi = np.log(psi)

            x_parent = sampled_values[parent]
            x_parent_idx = self.X.index(x_parent)
            # Compute unnormalized log-conditional distribution
            log_cond_prob = np.zeros(len(self.X))
            for idx, x_node in enumerate(self.X):
                log_edge_phi = log_psi[x_parent_idx, idx]
                log_cond_prob[idx] = (
                    log_phi_node[idx] + log_edge_phi + log_product_messages[idx]
                )

            # Normalize
            log_cond_prob -= logsumexp(log_cond_prob)
            cond_prob = np.exp(log_cond_prob)
            # Sample x_node directly
            x_node = np.random.choice(self.X, p=cond_prob)
            sampled_values[node] = x_node
            # Recursively sample children of node
            for neighbor in self.tree_graph.get_neighbors(node):
                if neighbor != parent:
                    sample_children(node, neighbor)

        for neighbor in tree_graph.get_neighbors(root):
            sample_children(root, neighbor)
        return sampled_values
