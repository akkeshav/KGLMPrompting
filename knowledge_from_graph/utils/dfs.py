import networkx as nx


def are_dest_concepts_visited(concepts, visited):
    for concept in concepts:
        if visited[concept] == 0:
            return False
    return True


def dfs(source, visited, adj, path, concepts):
    print("Source node -> ", source)
    for index, edge in enumerate(adj[source]):
        if edge == 1 and visited[index] == 0 and not are_dest_concepts_visited(concepts, visited):
            visited[index] = 1
            print("and len is ", len(path), "and element is ", index)
            path.append(str(source) + "relation_to" + str(index))
            dfs(index, visited, adj, path, concepts)


def dfs_reverse(source, visited, adj, path, concepts):
    if are_dest_concepts_visited(concepts, visited):
        return True
    for index, edge in enumerate(adj[source]):
        if edge == 1 and visited[index] == 0:
            visited[index] = 1
            print("and len is ", len(path), "and element is ", index)
            if dfs_reverse(index, visited, adj, path, concepts):
                path.append(str(source) + "relation_to" + str(index))
                return True


def main():
    G = nx.Graph()
    G.add_nodes_from([count for count in range(20)])
    G.add_edges_from([(1, 2), (1, 3), (3, 4), (3, 6), (4, 5), (6, 7)])
    print(G.number_of_nodes())
    print(G.number_of_edges())
    print(nx.adjacency_matrix(G))
    adj = nx.to_numpy_array(G)
    visited = [0 for ind in range(20)]
    visited[1] = 1
    path = []
    dfs_reverse(1, visited, adj, path, [1, 3, 7])
    print(visited)
    print(path)
    print(are_dest_concepts_visited([1, 3], visited))


if __name__ == '__main__':
    main()
