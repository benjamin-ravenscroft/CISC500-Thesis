from ete3 import Tree
import ete3
from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class AttributeClustering:
    def __init__(self):
        self.node_categories = {}
        self.t = Tree()
        return

    def initialize_tree(self):
        """Initializes ETE tree for storing defining cell type relationships
        """
        self.t = Tree()
        return
    
    def add_node(self, node_name, parent, class_category="Child"):
        """Adds node to tree under given parent, and assigns a class category to the node

        Args:
            node_name (str): Name for node, typically the cell phenotype
            parent (str): Name of parent node
            class_category (str, optional): Class for the node; this is used when determing dissimilarity between nodes. Defaults to "Child".
        """
        if parent==None:
            self.t.add_child(name=node_name)
        else:
            p = self.t&parent
            p.add_child(name=node_name)
        self.node_categories[node_name] = class_category

    def lowest_common_ancestor(self, node_a, node_b):
        """Finds the LCA between two nodes in the tree

        Args:
            node_a (str): Name of node_a
            node_b (str): Name of node_b

        Returns:
            str: Name of LCA node for node_a and node_b
        """
        a = self.t&node_a
        b = self.t&node_b
        return a.get_common_ancestor(b).name

    def get_node_class(self, node_name):
        """Retuns the class of a given node

        Args:
            node_name (str): Name of the node

        Returns:
            str: class category of node
        """
        return self.node_categories.get(node_name)

    #def dissimilarity(self, node_a, node_b):
    
    def get_distance_bw_nodes(self, node_a, node_b):
        """Function to find the distance between two nodes

        Args:
            node_a (str): Name of node_a
            node_b (str): Name of node_b

        Returns:
            float: distance between nodes, typically integer number of branches
        """
        #print("In")
        #print(node_a)
        #print(node_b)
        a = self.t&node_a
        b = self.t&node_b
        dist = self.t.get_distance(a, b)
        #print("Out")
        return dist
    
    def get_t1(self, G):
        """Get the dissimilarity threshold for determining if nodes' are spatially directly reachable

        Args:
            G (nx.graph): NetworkX graph object

        Returns:
            float: SDR threshold
        """
        adj_mtx = nx.to_numpy_matrix(G, nodelist=np.arange(len(G.nodes)))
        nn_d = np.array([])
        for p, row in enumerate(adj_mtx):
            nn = np.argmin(row)
            nn_d = np.append(nn_d, self.get_distance_bw_nodes(G.nodes[p]['class'], G.nodes[nn]['class']))
        # get rid of 3SD outliers
        std = np.std(nn_d)
        nn_d_red = nn_d[nn_d < (3*std)]
        t1 = np.mean(nn_d_red)
        # reporting
        #print(f"Full neighbour length: {len(nn_d)}\nOutliers removed neighbour length: {len(nn_d_red)}")
        return t1

    def get_density_indicators(self, G, t1):
        """Creates a list with density indicators of each node in Graph

        Args:
            G (nx.graph): NetworkX graph
            t1 (float): SDR threshold

        Returns:
            pd.DataFrame, np.array: dataframe containing pairs of node ID and density indicator, node adjacency matrix where SDR neighbours are marked
        """
        sdr_mtx = np.zeros((len(G.nodes), len(G.nodes)))
        n_mtx = sdr_mtx.copy()
        for i in G:
            for j in G[i]:
                if self.get_distance_bw_nodes(G.nodes[i]['class'], G.nodes[j]['class']) <= t1:
                    sdr_mtx[i, j] = 1
                    sdr_mtx[j, i] = 1
                n_mtx[i, j] = 1
        di = np.sum(sdr_mtx, axis=1) + np.divide(np.sum(sdr_mtx, axis=1), np.sum(n_mtx, axis=1))
        di = np.array([np.arange(len(G.nodes)).T, di.T])
        sorted_di = di[:, np.argsort(di[1,:])]
        di_df = pd.DataFrame(np.flip(sorted_di, axis=1)).T  # return as dataframe so that rows can be indexed using list of values
        di_df[di_df.columns[0]] = di_df.iloc[:, 0].astype(int)
        return di_df, sdr_mtx

    class Cluster:
        def __init__(self, AttributeClustering, sdr_mtx, di, G, t1):
            self.AttributeClustering = AttributeClustering
            self.nodetree = Tree()
            self.cluster_classes = {n_class: 0 for n_class in set(nx.get_node_attributes(G, "class").values())}
            self.sdr_mtx = sdr_mtx
            self.di = di
            self.G = G
            self.t1 = t1

        def add_node(self, parent, child):
            """Adds node to cluster tree under parent node (ie,, parental expanding core)

            Args:
                parent (int): node ID of expanding core
                child (int): node ID of expanding core neighbor being added to cluster
            """
            str_child = str(child)
            if parent==None:
                self.nodetree.add_child(name=str_child)
            else:
                parent_node = self.nodetree&str(parent)
                parent_node.add_child(name=str_child)
            n_class = self.G.nodes[child]['class']
            node = self.nodetree&str_child
            node.add_features(n_class=n_class)
            self.cluster_classes[n_class] += 1

        def get_avg_distance_to_CLU(self, n_class):
            """Gets the average dissimilarity between the class of the new node and the nodes already in the cluster

            Args:
                n_class (str): class of the new node

            Returns:
                float: average dissimilarity between new node and nodes already in cluster
            """
            # construct numpy array with distances to each class
            class_d = np.array([])
            for c in self.cluster_classes.keys():
                class_d = np.append(class_d, self.AttributeClustering.get_distance_bw_nodes(node_a=n_class, node_b=c))
            class_n = np.array(list(self.cluster_classes.values()))
            average = np.matmul(class_d, class_n)/np.sum(class_n)
            return average
    
        def add_expanding_core(self, exp_core):
            #print(f"Exp core: {exp_core}")
            sdr_nodes_unpruned = np.nonzero(self.sdr_mtx[exp_core])[0]  # get node's SDR neighbors
            sdr_nodes = []
            # prune nodes already in cluster
            for i,n in enumerate(sdr_nodes_unpruned):
                if str(n) not in self.nodetree:
                    sdr_nodes.append(n)

            new_nodes_counter = 0

            nn_di_df = self.di.loc[self.di[0].isin(sdr_nodes)]    # get their DIs in descending order
            #print(f"nn_di_df: {nn_di_df}")
            avg_distances = {}
            for node in nn_di_df.iloc[:,0]:
                #print(f"Node: {node}")
                n_class = self.G.nodes[node]['class']
                if n_class not in avg_distances:
                    avg_distances[n_class] = self.get_avg_distance_to_CLU(n_class)
                if avg_distances[n_class] <= self.t1:
                    self.add_node(exp_core, node)
                    new_nodes_counter += 1

            return new_nodes_counter

        def build_cluster(self):
            visited_nodes = []
            # add root node
            self.add_node(parent=None, child=self.di.loc[0,0])
            new_node_counter = 1
            while new_node_counter != 0:
                new_node_counter = 0
                for node in self.nodetree.traverse("levelorder"):
                    if (node.name not in visited_nodes) and (node.name != ''):
                        nodes_added = self.add_expanding_core(int(node.name))
                        visited_nodes.append(node.name)
                        new_node_counter += nodes_added

            return self.nodetree

        def build_cluster_v2(self, root_node, visited_nodes=[]):
            # add root node
            self.add_node(parent=None, child=root_node)
            new_node_counter = 1
            while new_node_counter != 0:
                new_node_counter = 0
                for node in self.nodetree.traverse("levelorder"):
                    if (node.name not in visited_nodes) and (node.name != ''):
                        nodes_added = self.add_expanding_core(int(node.name))
                        visited_nodes.append(node.name)
                        new_node_counter += nodes_added

            return self.nodetree

    def get_nodes_from_tree(self, nodetree):
            cluster_nodes = []
            for node in nodetree.traverse("levelorder"):
                if node.name != '':
                    cluster_nodes.append(int(node.name))
            return cluster_nodes
        
    def build_clusters(self, sdr_mtx, di, G, t1):
        max_nodes = di.shape[0]
        cluster_nodes_list = []
        visited_nodes = []
        total = 0

        # start at index 0 of density indicator dataframe for first cluster
        while total < max_nodes:
            c = self.Cluster(AttributeClustering=self, sdr_mtx=sdr_mtx, di=di, G=G, t1=t1)
            start_node = di[~di[0].isin(visited_nodes)].iloc[0,0]
            nodetree = c.build_cluster_v2(start_node, visited_nodes)
            nodes = self.get_nodes_from_tree(nodetree)
            cluster_nodes_list.append(nodes)
            total += len(nodes)
            visited_nodes += nodes
            visited_nodes = list(set(visited_nodes))

        return cluster_nodes_list

    def merge_clusters(self, clusters, threshold):
        """Function to merge clusters whose size falls below a given threshold.


        Args:
            clusters (list of lists): list of clusters (each cluster is a list of nodes that belong to said cluster)
            threshold (int): size threshold for clusters. If len(cluster) > threshold -> cluster is considered to be genuine. If len(cluster) < threshold, cluster is considered to be noise and inconsequential.

        Returns:
            list of lists: list of clusters that are considered genuine (ie,, length > threshold)
        """
        out_clusters = clusters.copy()
        merged_clusters = []
        for c in clusters:
            if len(c) < threshold:
                merged_clusters.append(c)
                out_clusters.remove(c)
        return out_clusters

    def assign_nodes_clusters(self, G, clusters):
        clust_arr = np.zeros(len(G.nodes))
        for i in range(len(clusters)):
            clust_arr[clusters[i]] = i+1
        return clust_arr

    @staticmethod
    def save_graph(G, path):
        """Save graph in GraphML format using LXML framework (faster than XML)

        Args:
            G (networkx graph): Graph
            path (str): path for to save graph to (without suffix filetype)
        """
        temp_G = G.copy()
        for node in list(temp_G.nodes):
            p = temp_G.nodes[node]['point']
            temp_G.nodes[node]['x'] = p[0]
            temp_G.nodes[node]['y'] = p[1]
            del temp_G.nodes[node]['point']
        nx.write_graphml_lxml(temp_G, path+".graphml")

    @staticmethod
    def read_graph(path):
        G = nx.read_graphml(path)
        for node in list(G.nodes):
            G.nodes[node]['point'] = [G.nodes[node]['x'], G.nodes[node]['y']]
            del G.nodes[node]['x']
            del G.nodes[node]['y']
        return G

    @staticmethod
    def draw_graph_w_clusters(G, figsize=(10,10), clusters=None):
        if clusters is not None:
            cmap = plt.get_cmap('nipy_spectral')
            cols = []
            n_cols = len(set(clusters))
            for i,_ in enumerate(set(clusters)):
                cols.append(cmap((1/n_cols)*i))

        nodes = dict(G.nodes.data())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        if clusters is not None:
            for p in nodes:
                ax.scatter(nodes[p]['point'][0], nodes[p]['point'][1], color=cols[int(clusters[p])])
        else:
            for p in nodes:
                ax.scatter(nodes[p]['point'][0], nodes[p]['point'][1])

        edges = G.edges
        for e in edges:        
            a = FancyArrowPatch(nodes[e[0]]['point'], nodes[e[1]]['point'])
            ax.add_artist(a)
        
        plt.draw()
        plt.show()
 