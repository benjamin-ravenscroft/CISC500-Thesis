import numpy as np
from scipy.spatial import distance_matrix, Delaunay
from itertools import combinations, product
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

class SpatialStatistics:
    def __init__(self):
        return
    
    def moran_i(self, w, x):
        """Calculates global Moran's I

        Args:
            w (2d-array or matrix): 2D distance matrix containing the spatial weights for all spatial units
            x (1d-arry or list): measurement of the variable of interest for each spatial unit
        """
        x_mean = np.mean(x)
        N = len(x)
        W = np.sum(w)
        x_sub = np.subtract(x, x_mean)
        num_mat = np.asarray([[(p1*p2) for p2 in x_sub] for p1 in x_sub])

        numerator = np.sum(np.multiply(w, num_mat))
        denominator = np.sum(np.multiply(x_sub, x_sub))

        m_i = (N/W)*(numerator/denominator)
        
        return m_i

    def delaunay_triangulation(self, points):
        """Generates a Delaunay triangulation for set of points

        Args:
            points (2d-array): 2d-array containing x, y coordinates of points
        """
        tri = Delaunay(points)
        return tri
    
    def plot_delaunay(self, tri, figsize=(10,10)):
        """Plots the Delaunay triangulation

        Args:
            tri (scipy.spatial.Delaunay object): Triangulation object with simplices and points
        """
        fig = plt.figure(figsize=figsize)
        plt.triplot(tri.points[:,0], tri.points[:,1], tri.simplices)
        plt.plot(tri.points[:,0], tri.points[:,1], 'o')
        plt.show()

    def create_distance_matrix(self, points, p=2):
        dmat = distance_matrix(points, points, p=2)
        return dmat

    def _set_distance_matrix(self, dist_mat):
        """Method for setting class distance matrix

        Args:
            dist_mat (2d-array): 2d distance matrix
        """
        self.dmat = dist_mat

    def graph_from_delaunay(self, tri, dist_mat):
        G = nx.Graph()  # initialize empty NetowrkX graph
        # iterate through simplices and add edges
        count = 0
        for verts in tri.simplices:
            pairs = list(combinations(verts, 2))
            for pair in pairs:
                G.add_node(pair[0], point=tri.points[pair[0]])
                G.add_node(pair[1], point=tri.points[pair[1]])
                G.add_edge(pair[0], pair[1], weight=dist_mat[pair[0], pair[1]])
            count += 1

        #print(sorted(list(G.nodes())))
        print(f"Number of nodes: {len(G)}")
        
        return G

    def get_global_graph_mean_std(self, G):
        flat_arr = nx.to_numpy_array(G).ravel()
        nonzero_inds = np.nonzero(flat_arr)
        std = np.std(flat_arr[nonzero_inds])
        mean = np.mean(flat_arr[nonzero_inds])
        return mean, std

    def get_local_graph_mean(self, G):
        adj_mtx = nx.to_numpy_array(G, nodelist=np.arange(len(G.nodes())))
        nonzero_row = (adj_mtx != 0).sum(1)
        row_sum = np.sum(adj_mtx, axis=1)
        mean = np.divide(row_sum, nonzero_row)
        return mean


    def get_global_distance_contraint_matrix(self, G):
        mean, std = self.get_global_graph_mean_std(G)
        alpha = mean/self.get_local_graph_mean(G)
        gdc_list = mean + np.multiply(alpha, std)

        gdc_arr = nx.to_numpy_array(G, nodelist=np.arange(len(G.nodes())))
        for p in range(gdc_arr.shape[0]):
            gdc_arr[np.nonzero(gdc_arr[p])] = gdc_list[p]

        return gdc_arr
    
    def reduce_graph_global_constraint(self, G):
        red_G = G.copy()
        gdc_arr = self.get_global_distance_contraint_matrix(G)
        g_arr = nx.to_numpy_array(G, nodelist=np.arange(len(G.nodes())))
        nodes = np.arange(len(G.nodes()))

        for p in range(g_arr.shape[0]):
            inds = np.where(g_arr[p] > gdc_arr[p], True, False)
            remove_nodes = nodes[inds]
            remove_edges = product([p], remove_nodes)
            red_G.remove_edges_from(remove_edges)

        return red_G

    def draw_graph(self, G, figsize=(10,10), clusters=None):
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

    def get_n_order_paths(self, G, u, n):
        if n == 0:
            return [[u]]
        paths = []
        for neighbor in G.neighbors(u):
            for path in self.get_n_order_paths(G, neighbor, n-1):
                #if u not in path:
                paths.append([u]+path)
        return paths
        
    def get_nodes_from_path(self, paths):
        nodes = list(set([y for x in paths for y in x]))
        return nodes

    def get_edges_from_path(self, paths):
        edges = []
        for path in paths:
            for i in range(1, len(path)):
                edge = sorted((path[i-1], path[i]))
                if edge not in edges:
                    edges.append(edge)
                    
        return edges

    def get_average_edge_length(self, edges, adj_mtx):
        edge_sum = 0

        for edge in edges:
            edge_sum += adj_mtx[edge[0], edge[1]]
        
        return edge_sum/len(edges)

    def get_n_order_mean(self, G, order):
        nodelist = np.arange(len(G.nodes()))
        
        means = np.array([])
        for node in nodelist:
            paths = self.get_n_order_paths(G, node, order)
            edges = self.get_edges_from_path(paths)
            avg_len = self.get_average_edge_length(G, edges)
            means = np.append(means, avg_len)
        
        return means

    def get_node_local_variation(self, G, node, adj_mtx):
        edges = np.array([])
        for neighbour in G.neighbors(node):
            edges = np.append(edges, adj_mtx[node, neighbour])
        return np.std(edges)

    def get_n_order_local_variation(self, G, order=2):
        nodelist = np.arange(len(G.nodes()))

        local_var_list = []
        for node in nodelist:
            paths = self.get_n_order_paths(G, node, order)
            nodes = self.get_nodes_from_path(paths)
            var_arr = np.array([])
            for n in nodes:
                var_arr = np.append(var_arr, self.get_node_local_variation(G, n))
            local_var_list.append(np.mean(var_arr))
        
        return local_var_list

    def get_local_distance_contraint(self, G, beta, order):
        nodelist = np.arange(len(G.nodes()))
        adj_mtx = nx.to_numpy_array(G, nodelist=nodelist)
        local_var_arr = np.array([0]*len(nodelist))
        checked_nodes = []
        n_order_nodes = [[]]
        ldc = []

        means = np.array([])
        for node in nodelist:
            paths = self.get_n_order_paths(G, node, order)
            # compute the local 2-order-mean
            edges = self.get_edges_from_path(paths)
            if len(edges) != 0:
                avg_len = self.get_average_edge_length(edges, adj_mtx)
            else:
                avg_len = 0
            means = np.append(means, avg_len)
            # get local variation
            nodes = self.get_nodes_from_path(paths)
            n_order_nodes.append(nodes)
            new_nodes = set(nodes) - set(checked_nodes) # only compute var for unseen nodes
            for n in new_nodes:
                local_var_arr[n] = self.get_node_local_variation(G, n, adj_mtx)
                checked_nodes.append(n)
            mean_var = np.sum(local_var_arr[nodes])/len(nodes)
            ldc.append(avg_len + (beta * mean_var))
        
        return ldc, n_order_nodes

    def reduce_graph_local_constraint(self, G, ldc_l, n_order_nodes, adj_mtx, beta=2, order=2):
        red_G = G.copy()
        nodelist = np.arange(len(G.nodes()))
        for node in nodelist:
            ldc = ldc_l[node]
            nodes = n_order_nodes[node]
            del_edges = []
            for n in nodes:
                del_nodes = nodelist[adj_mtx[n] > ldc]
                edges = [tuple(sorted(x)) for x in list(product([n], del_nodes))]
                del_edges = del_edges + edges
                
        #print(del_edges)
        red_G.remove_edges_from(set(del_edges))

        return red_G

    def add_class_to_graph(self, G, classes):
        new_G = G.copy()
        for i in range(len(new_G.nodes)):
            new_G.nodes[i]['class'] = classes[i]

        return new_G

    def add_cluster_to_graph(self, G, clusters):
        """Add cluster number to graph nodes as an attribute

        Args:
            G (networkx graph): graph
            cluster (np.array): numpy array with node clusters as integers
        """
        new_G = G.copy()
        for i in range(len(new_G.nodes)):
            new_G.nodes[i]['cluster'] = clusters[i]