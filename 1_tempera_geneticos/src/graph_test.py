
import random
import networkx as nx
import matplotlib.pyplot as plt

# size - tamanho do grafo
# weight range - intervalo de numeros em que se encontram os pesoas das arestas
def get_connected_graph(size, weight_range=(1,100)):
    G = nx.complete_graph(size)
    
    for u, v in G.edges():
        G.edges[u, v]['weight'] = random.randint(*weight_range)
        
    removabel_edges = list(G.edges())
    
    for edge in removabel_edges:
        u, v = edge
        G.remove_edge(u, v)
        
        if not nx.is_connected(G):
            G.add_edge(u, v, weight_range=random.randint(*weight_range))
    
    return G

def get_complete_graph(size, weight_range=(1,100)):
    G = nx.complete_graph(size)
    
    for u, v in G.edges():
        G.edges[u, v]['weight'] = random.randint(*weight_range)    
    
    return G

# G - grafo
# tour - caminho
# current_node - nodulo atual
# pos - dicionario com lista de adjacencias do grafo
def plot_graph_step(G, tour, current_node, pos):
    plt.clf() # clear
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    path_edges = list(zip(tour, tour[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='green', node_size=500)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.pause(0.5)
    
def calculate_tour_cost(G, tour):
    return sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour) - 1))

# G - grafo
# start node - nodulo inicial
def nearest_neighbor_tsp(G, start_node=None):
    if start_node is None:
        start_node = random.choice(list(G.nodes))
    
    # pos = nx.spring_layout(G)
    # plt.ion()
    # plt.show()
    
    # heuristica
    
    unvisited = set(G.nodes)
    unvisited.remove(start_node)
    tour = [start_node]
    current_node = start_node
    
    # plot_graph_step(G, tour, current_node, pos)

    while unvisited:
        # escolha gulosa -> vizinho mais proximo
        next_node = min(unvisited, key=lambda node: G[current_node][node]['weight'])
        unvisited.remove(next_node)
        tour.append(next_node)
        current_node = next_node
        # plot_graph_step(G, tour, current_node, pos)
    
    tour.append(start_node)
    # plot_graph_step(G, tour, current_node, pos)
    
    print(tour)
    tour_cost = calculate_tour_cost(G, tour)
    print(f'tour cost = {tour_cost}')
    
    # plt.ioff()
    # plt.show()

def random_neighbor_tsp(G, start_node=None):

    if start_node is None:
        start_node = random.choice(list(G.nodes))
    unvisited = set(G.nodes)
    unvisited.remove(start_node)
    tour = [start_node]
    current_node = start_node
    
    while unvisited:
        # arestas da vizinhanca do nodulo atual
        edges = [node for node in unvisited if node in G[current_node]]
        # escolha aleatoria
        if edges:
            next_node = random.choice(edges)
        else:
            next_node = None

        unvisited.remove(next_node)
        tour.append(next_node)
        current_node = next_node
        
    tour.append(start_node)
    print(tour)
    tour_cost = calculate_tour_cost(G, tour)
    print(f'tour cost = {tour_cost}')
    

# def main():
#     G = get_complete_graph(5)
#     for i in range(0, 1):
#         nearest_neighbor_tsp(G, 0)
#         random_neighbor_tsp(G, 0)
