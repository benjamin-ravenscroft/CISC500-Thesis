import networkx as nx

class ClusterAttributes:
    def __init__(self, cluster_nodes):
        self.cluster_nodes = cluster_nodes
        self.visited_nodes = []
        return
    
    def get_n_order_paths(self, G, u, n):
        if n == 0:
            return [[u]]
        paths = []
        # get the neighbors and remove all nodes that have already been visited as well as the nodes in the cluster
        neighbors = set(G.neighbors(u))
        neighbors.difference_update(set(self.visited_nodes))
        neighbors.difference_update(set(self.cluster_nodes))

        for neighbor in neighbors:
            for path in self.get_n_order_paths(G, neighbor, n-1):
                #if u not in path:
                paths.append([u]+path)
        return paths
        
    def get_nodes_from_path(self, paths):
        nodes = list(set([y for x in paths for y in x]))
        self.visited_nodes += nodes # add to the visited nodes so that they aren't revisited
        return nodes

    def get_cluster_n_order_neighborhood(self, G, n):
        n_neighborhood_nodes = []
        for node in self.cluster_nodes:
            paths = self.get_n_order_paths(G, node, n)
            nodes = list(set(self.get_nodes_from_path(paths)).difference(set(self.cluster_nodes)))
            n_neighborhood_nodes += nodes
        return n_neighborhood_nodes
