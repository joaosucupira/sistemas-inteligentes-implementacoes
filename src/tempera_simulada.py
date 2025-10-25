import networkx as nx
import matplotlib.pyplot as plt
import random 
import math 
import numpy as np
import time

class Simulated_Annealing():
    def __init__(self, n, T_initial, T_finish, alpha):
        # Parâmetros para a têmpera simulada
        self.n = n
        self.T_initial = T_initial
        self.T_finish = T_finish
        self.alpha = alpha

        self.graph_tsp = self.create_graph(self.n)
        self.current_route = list(range(self.n))
        random.shuffle(self.current_route)


    def create_graph(self, n):
        G = nx.complete_graph(n)
        
        # Adicionar posições aleatórias para as cidades (para visualização)
        pos = {i: (random.uniform(0, 100), random.uniform(0,100)) for i in range(n)}
        nx.set_node_attributes(G, pos, 'pos')

        # Adicionar pesos (distâncias) às arestas
        for i,j in G.edges():
            dist = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
            G[i][j]['weight'] = dist 
        
        return G
    
    def cost_calculate(self, route, graph):
        """
        Função Objetiva: Calcula o custo total da rota.
        Reflexo da modelagem de 'custo' a ser minimizado.
        """
        cost = 0 
        
        for i in range(len(route) - 1):
            cost += graph[route[i]][route[i+1]]['weight']
        cost += graph [route[-1]][route[0]]['weight']  # Retorno ao ponto inicial

        return cost
    
    def generate_2opt(self, route):
        """
        Função de Vizinhança: Gera uma nova rota 'vizinha' usando a técnica 2-opt.
        Reflete a navegação pelo 'espaço de soluções'.
        """
        new_route = route[:]
        i,j = sorted(random.sample(range(1, len(new_route)),2))
        new_route[i:j] = reversed(new_route[i:j])

        return new_route
    
    def tempera_simulada(self):
        """
        Implementação central da Têmpera Simulada.
        """
        current_route = self.current_route[:]
        
        better_route = current_route[:]
        better_cost = self.cost_calculate(better_route, self.graph_tsp)

        iteraction_cost = []
        T = self.T_initial

        while T > self.T_finish:
            neighbor_route = self.generate_2opt(current_route)
            neighbor_cost = self.cost_calculate(neighbor_route, self.graph_tsp)
            current_cost = self.cost_calculate(current_route, self.graph_tsp)

            delta_cost = neighbor_cost - current_cost

            # Probabilidade de aceitação (analogia termodinâmica)
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_route = neighbor_route

            # Atualiza a melhor rota encontrada até o momento
            if self.cost_calculate(current_route, self.graph_tsp) < better_cost:
                better_route = current_route[:]
                better_cost = self.cost_calculate(better_route, self.graph_tsp)

            iteraction_cost.append(better_cost)

            T *= self.alpha # Decaimento da temperatura

        return better_route, better_cost, iteraction_cost

    def plot_route(self, route, title):
        """
        Plota o grafo e a rota final usando Matplotlib e NetworkX.
        """
        pos = nx.get_node_attributes(self.graph_tsp, 'pos')

        plt.figure(figsize=(10,8))

        nx.draw_networkx_nodes(self.graph_tsp, pos, node_size=200, node_shape="o", node_color='skyblue')

        # Desenhar arestas da rota
        edges_route = [(route[i], route[i+1]) for i in range(len(route)-1)]
        edges_route.append((route[-1], route[0]))
        nx.draw_networkx_edges(self.graph_tsp, pos, edgelist=edges_route, edge_color='red', width=2)

        nx.draw_networkx_labels(self.graph_tsp, pos)
        plt.title(f"{title}\nTotal Cost: {self.cost_calculate(route, self.graph_tsp):.2f}")
        plt.show()
   
    def main(self):
        start_time = time.time()
        better_route, better_cost, iteraction_cost = self.tempera_simulada()
        end_time = time.time()
        exec_time = end_time - start_time

        print(f"___Simulated Annealing___")
        print(f"Number of cities: {self.n}")
        print(f"Better Route Founded: {better_route}")
        print(f"Final Cost: {better_cost:.2f}")
        print(f"Execution Time: {exec_time:.4f} seconds")

        self.plot_route(route=better_route, title="Final route found by the simulated annealing")
        return iteraction_cost, exec_time, better_cost

# Testes e Análises Múltiplas
if __name__ == "__main__":
    
    # Análise de Convergência e Impacto dos Parâmetros 
    print("\n##### Análise Comparativa de Convergência ####")
    test_params_convergency = [
        {'n': 20, 'T_initial': 10000, 'T_finish': 1, 'alpha': 0.99, 'label': 'Alpha=0.99 (Rápido)'},
        {'n': 20, 'T_initial': 10000, 'T_finish': 1, 'alpha': 0.999, 'label': 'Alpha=0.999 (Lento)'},
        {'n': 20, 'T_initial': 100, 'T_finish': 1, 'alpha': 0.99, 'label': 'T_inicial Baixa'}
    ]

    all_costs = {}
    
    # Cria um grafo novo e fixo para todos os testes comparativos
    fixed_graph_20_cities = Simulated_Annealing(20, 0, 0, 0).create_graph(20)
    
    for params in test_params_convergency:
        tsp_test = Simulated_Annealing(params['n'], params['T_initial'], params['T_finish'], params['alpha'])
        tsp_test.graph_tsp = fixed_graph_20_cities
        
        print(f"\nIniciando teste com: {params['label']}")
        iteraction_costs, _, _ = tsp_test.main()
        all_costs[params['label']] = iteraction_costs

    plt.figure(figsize=(12, 8))
    for label, costs in all_costs.items():
        plt.plot(costs, label=label)

    plt.title("Análise Comparativa de Convergência da Têmpera Simulada")
    plt.xlabel("Iterações")
    plt.ylabel("Custo da Melhor Rota")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()
    
    print("\n#### Análise de Tempo de Execução vs. Número de Cidades ####")
    cities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    exec_times = []
    final_costs = []

    for n in cities:
        print(f"\nTestando com {n} cidades...")
        tsp_solver = Simulated_Annealing(n=n, T_initial=10000, T_finish=1, alpha=0.99)
        _, exec_time, final_cost = tsp_solver.main()
        exec_times.append(exec_time)
        final_costs.append(final_cost)

    plt.figure(figsize=(12, 8))
    plt.plot(cities, exec_times, marker='o', linestyle='-', color='b')
    plt.title("Tempo de Execução vs. Número de Cidades (Têmpera Simulada)")
    plt.xlabel("Número de Cidades (n)")
    plt.ylabel("Tempo de Execução (s)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(cities, final_costs, marker='o', linestyle='-', color='r')
    plt.title("Custo Final vs. Número de Cidades (Têmpera Simulada)")
    plt.xlabel("Número de Cidades (n)")
    plt.ylabel("Custo da Melhor Rota")
    plt.grid(True)
    plt.show()
    