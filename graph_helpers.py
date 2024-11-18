import numpy as np
from graphviz import Graph
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
    
    def visualize(self, output_file=None, show_potentials=False):
        """
        Visualize the graph using Graphviz.

        Args:
            output_file (str): If provided, saves the visualization to a file (e.g., "graph.png").
            show_potentials (bool): Whether to annotate nodes and edges with their potentials.
        """
        dot = Graph(format='png', engine='dot')
        dot.attr('node', shape='circle')

        # Add nodes
        for node in self.get_V():
            if show_potentials:
                potential = self.node_potentials[node]
                label = f"{node}\n{potential}"
            else:
                label = f"{node}"
            dot.node(str(node), label=label)

        # Add edges
        for edge in self.get_E():
            if show_potentials:
                potential = self.edge_potentials[edge]
                label = f"{potential}"
            else:
                label = ""
            dot.edge(str(edge[0]), str(edge[1]), label=label)

        # Render the graph
        if output_file:
            dot.render(output_file, view=True)  # Saves and optionally opens the file
        else:
            dot.view()  # Only opens the visualization
        
class Ising2D:
    '''generate a simple 2D Ising Graph according to P_xv = 1/Z exp{sum{theta*x_i*x_j}}'''
    def __init__(self, N, theta,X =[-1,1]):
        '''
        N: size of ising graph (N*N nodes)
        theta: coupling parameter
        X: an array of possible x values
        '''
        self.N = N
        self.theta = theta
        self.X = X
        self.Ising = self.generate_2Dising(N,theta,X)
    
    def generate_2Dising(self, N, theta, X):
        '''
        Generate a 2D Ising model graph using the GraphModel class.
        '''
        node_potentials = {}
        edge_potentials = {}
        
        # Define node potentials (assume uniform distribution over X for simplicity)
        for i in range(N):
            for j in range(N):
                node_index = (i, j)  # Node represented by its 2D coordinates
                node_potentials[node_index] = np.array([1, 1])  # Uniform potential for each node

        # Define edge potentials based on the Ising model
        for i in range(N):
            for j in range(N):
                node_index = (i, j)

                # Add edges to the right and below (avoiding duplicates)
                if j + 1 < N:  # Horizontal neighbor
                    neighbor_index = (i, j + 1)
                    x0 = self.X[0]
                    x1 = self.X[1]
                    edge_potentials[(node_index, neighbor_index)] = np.array([
                        [np.exp(theta*x0*x0), np.exp(theta*x0*x1)],
                        [np.exp(theta*x1*x0), np.exp(theta*x1*x1)]
                    ])

                if i + 1 < N:  # Vertical neighbor
                    neighbor_index = (i + 1, j)
                    x0 = self.X[0]
                    x1 = self.X[1]
                    edge_potentials[(node_index, neighbor_index)] = np.array([
                        [np.exp(theta*x0*x0), np.exp(theta*x0*x1)],
                        [np.exp(theta*x1*x0), np.exp(theta*x1*x1)]
                    ])

        # Create a GraphModel instance with node and edge potentials
        return GraphModel(node_potentials, edge_potentials)
    
    def split_and_calculate_trees(self):
        '''
        Split the Ising graph into disjoint sets A and B, and calculate the conditional potentials
        for P(A | B) and P(B | A).
        Returns:
            tree_A: GraphModel object representing P(A | B)
            tree_B: GraphModel object representing P(B | A)
        '''
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
                if i == 0 or (j % 2 == 0 and i <self.N-1):  # First row or odd column -> Tree A
                    nodes_A.add(node_index)
                if i == self.N - 1 or (j % 2 == 1 and i>0):  # Last row or even column -> Tree B
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
    def calculate_tree_potentials(self,tree_nodes, other_tree_nodes, edges, theta):
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
                            phi += theta*x_i*x_j
                potential.append(np.exp(phi))
            node_potentials[node] = np.array(potential)

        # Calculate edge potentials
        for edge in edges:
            edge_potentials[edge] = self.Ising.edge_potentials[edge]

        return node_potentials, edge_potentials

    



class TreeSampler:
    '''implement tree sampler to sample from joint distribution of a tree'''
    def __init__(self, node_potentials, edge_potentials):
        self.tree_graph = GraphModel(node_potentials,edge_potentials)

    def compute_message(self,i, j, messages):

        '''
        Takes in a GraphModel object representing a tree graph and computes the
        corresponding partition function. 
        return (Z, messages) where
        Z: float equal to the partition function (sum over all nodes)
        messages: map from (i,j) to message data structure  
        representing message passed from node i to node j.
        '''
        tree_graph = self.tree_graph

        # Check if the message (i, j) is already computed
        if (i, j) in messages:
            return messages[(i, j)]

        # Node potential for node i
        phi_i = tree_graph.node_potentials[i]

        # Initialization
        message_ij = np.ones(2)

        # Get the neighbors of node i
        neighbors_i = [
            neighbor for neighbor in tree_graph.get_neighbors(i) if neighbor != j
        ]

        e_ij = (i, j) if (i, j) in tree_graph.edge_potentials else (j, i)
        psi_ij = tree_graph.edge_potentials[e_ij]

        if not neighbors_i:
            message_ij = phi_i * psi_ij
            if e_ij == (i, j):
                message_ij = np.matmul(phi_i, psi_ij)
            else:
                message_ij = np.matmul(phi_i, psi_ij.T)
        else:
            # If i has other neighbors (let's call them k), recursively compute messages from k to i
            product_messages = np.ones(2)
            for neighbor in neighbors_i:
                message_ki = self.compute_message(neighbor, i, messages, tree_graph)
                product_messages *= message_ki

                if e_ij == (i, j):
                    message_ij = np.matmul(product_messages * phi_i, psi_ij)
                else:
                    message_ij = np.matmul(product_messages * phi_i, psi_ij.T)

        # Store the computed message (i, j)
        messages[(i, j)] = message_ij

        return message_ij


    def sum_product_partition(self):
        tree_graph = self.tree_graph
        messages = {}
        nodes = tree_graph.get_V()
        root = nodes[-1]

        # bottom up
        for neighbor in tree_graph.get_neighbors(root):
            self.compute_message(neighbor, root, messages, tree_graph)

        # top down
        def top_down_message(parent, node, messages):
            # Compute message from parent to child (node)
            self.compute_message(parent, node, messages, tree_graph)

            # Recursively propagate messages to children of 'node'
            for neighbor in tree_graph.get_neighbors(node):
                if neighbor != parent:
                    top_down_message(node, neighbor, messages)

        for neighbor in tree_graph.get_neighbors(root):
            top_down_message(root, neighbor, messages)

        phi_root = tree_graph.node_potentials[root]
        Z = 0
        for x_root in [0, 1]:
            root_contribution = phi_root[x_root]
            for neighbor in tree_graph.get_neighbors(root):
                root_contribution *= messages[(neighbor, root)][x_root]
            Z += root_contribution
        return Z, messages