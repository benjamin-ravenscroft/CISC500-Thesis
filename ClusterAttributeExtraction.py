from itertools import combinations
import pandas as pd
from ClusterAttributes import ClusterAttributes
from itertools import product
import numpy as np

class ClusterAttributeExtraction:
    def __init__(self, cell_types, clusters, G):
        self.cell_types = cell_types
        self.clusters = clusters
        self.G = G

    def create_cluster_type_indice_dict(self):
        # possible cluster types -> product of cell types (alphabetically ordered sublists)
        combos = list(combinations(self.cell_types, 7))
        combos = combos + list(combinations(self.cell_types, 6))
        combos = combos + list(combinations(self.cell_types, 5))
        combos = combos + list(combinations(self.cell_types, 4))
        combos = combos + list(combinations(self.cell_types, 3))
        combos = combos + list(combinations(self.cell_types, 2))
        for i, sublist in enumerate(combos):
            combos[i] = ','.join(sorted(sublist))
        combos = self.cell_types + combos

        # get number of clusters
        self.cluster_type_inds = dict.fromkeys(combos, [])
        print(self.cluster_type_inds)

    def get_cluster_type_indices(self):
        cluster_types = []
        for i, c in enumerate(self.clusters):
            l = [dict(self.G.nodes(data="class"))[x] for x in c]
            t = ','.join(sorted(list(set(l))))
            self.cluster_type_inds[t] = self.cluster_type_inds[t] + [i]

    def get_summary_data_on_cluster_type(self, cluster_type, red_class_G):
        neighbor_cell_type_cols = [cluster_type+'.N.'+x for x in self.cell_types]
        clust_inds = self.cluster_type_inds[cluster_type]
        c_type_df = pd.DataFrame(columns=['C.'+cluster_type]+neighbor_cell_type_cols, data=np.zeros((len(clust_inds), len(self.cell_types)+1)))
        for r,i in enumerate(clust_inds):
            cluster = self.clusters[i]
            c_type_df.loc[r, 'C.'+cluster_type] = len(cluster)
            ca = ClusterAttributes(cluster_nodes=cluster)
            neighborhood_nodes = ca.get_cluster_n_order_neighborhood(self.G, 2)
            classes = [dict(red_class_G.nodes(data="class"))[key] for key in neighborhood_nodes]
            for c in classes:
                c_n = cluster_type+'.N.'+c
                c_type_df.loc[r, c_n] += 1

        summary_cols = ['Mean', 'Median', 'Min', 'Max']
        cols = list(product(summary_cols, ['C.'+cluster_type]+neighbor_cell_type_cols))
        for i,o in enumerate(cols):
            cols[i] = '.'.join(o)
        summ_df = pd.DataFrame(columns=['Num.C.'+cluster_type]+cols, data=np.zeros((1, len(cols)+1)))

        for i in c_type_df.columns:
            summ_df["Mean."+i] = c_type_df[i].mean()
            summ_df["Median."+i] = c_type_df[i].median()
            summ_df["Min."+i] = c_type_df[i].min()
            summ_df["Max."+i] = c_type_df[i].max()
        summ_df['Num.C.'+cluster_type] = len(clust_inds)
    
        return summ_df

    def create_cluster_attribute_df(self):
        # extract cluster attributes
        self.create_cluster_type_indice_dict()  # create dictionary to hold indices of the different cluster types
        self.get_cluster_type_indices() # add the indices to the dictionary

        # create a dataframe to hold data on each cluster
        # columns: cluster cells|neighborhood 1 cells|neighborhood 2 cells|etc.
        out_df = pd.DataFrame()
        for c in self.cluster_type_inds.keys():
            temp_df = self.get_summary_data_on_cluster_type(c, self.G)
            out_df = pd.concat([out_df, temp_df], axis=1)

        return out_df