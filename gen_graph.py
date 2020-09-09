import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

adj_mats = []
deg_mats = []
lap_mats = []
edge_mats = []
adj_list_mats = []

G = nx.Graph()
count = 0
num_nodes = 10
while count < 1000:  # Set the number of generated graphs
    G = nx.fast_gnp_random_graph(num_nodes, 0.4)
    if nx.is_eulerian(G): #generated Eulerian graphs
    # if not nx.is_eulerian(G): # generated non-Eulerian graphs
        adj_mat = np.array(nx.adj_matrix(G).todense())  # Adjacency matrix
        print(adj_mat)

        deg_mat = np.zeros_like(adj_mat)  # Degree matrix
        for i in range(adj_mat.shape[0]): deg_mat[i][i] = adj_mat[i].sum()  # The number on the diagonal is the sum of each row of the adjacency matrix
        print(deg_mat)

        lap_mat = deg_mat - adj_mat # Laplacian matrix
        print(lap_mat)

        edge_mat = np.array([[i[0], i[1]] for i in nx.to_edgelist(G)])  # Edge list
        print(nx.to_edgelist(G))
        print(edge_mat)

        adj_list_mat = [v for k, v in nx.to_dict_of_lists(G).items()]  # Adjacency list
        for i in range(len(adj_list_mat)):
            for j in range(9 - len(adj_list_mat[i])):
                adj_list_mat[i].append(0)
        adj_list_mat = np.array(adj_list_mat)
        print(nx.to_dict_of_lists(G))
        print(adj_list_mat)


        count += 1

        adj_mats.append(adj_mat)
        deg_mats.append(deg_mat)
        lap_mats.append(lap_mat)
        edge_mats.append(edge_mat)
        adj_list_mats.append(adj_list_mat)
        # print(count)
        nx.draw(G, with_labels=True)
        plt.show()
        break

    G.clear()

adj_mats = np.array(adj_mats)
deg_mats = np.array(deg_mats)
lap_mats = np.array(lap_mats)
edge_mats = np.array(edge_mats)
adj_list_mats = np.array(adj_list_mats)

# np.save('data/train_eulerian_adj.npy', adj_mats)
# np.save('data/train_eulerian_deg.npy', deg_mats)
# np.save('data/train_eulerian_lap.npy', lap_mats)
# np.save('data/train_eulerian_edge.npy', edge_mats)
# np.save('data/train_eulerian_adj_list.npy', adj_list_mats)


# np.save('data/eval_eulerian_adj.npy', adj_mats)
# np.save('data/eval_eulerian_deg.npy', deg_mats)
# np.save('data/eval_eulerian_lap.npy', lap_mats)
# np.save('data/eval_eulerian_edge.npy', edge_mats)
# np.save('data/eval_eulerian_adj_list.npy', adj_list_mats)


# np.save('data/train_non_eulerian_adj.npy', adj_mats)
# np.save('data/train_non_eulerian_deg.npy', deg_mats)
# np.save('data/train_non_eulerian_lap.npy', lap_mats)
# np.save('data/train_non_eulerian_edge.npy', edge_mats)
# np.save('data/train_non_eulerian_adj_list.npy', adj_list_mats)


# np.save('data/eval_non_eulerian_adj.npy', adj_mats)
# np.save('data/eval_non_eulerian_deg.npy', deg_mats)
# np.save('data/eval_non_eulerian_lap.npy', lap_mats)
# np.save('data/eval_non_eulerian_edge.npy', edge_mats)
# np.save('data/eval_non_eulerian_adj_list.npy', adj_list_mats)
