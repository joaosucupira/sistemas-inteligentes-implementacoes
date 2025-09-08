import matplotlib.pyplot as plt
from sys import maxsize
from random import randint, random, choice, shuffle, uniform
from string import ascii_uppercase

INF = maxsize
GRAPH = 100
WEIGHT_RANGE = (1,100)
MUTATION = 0.4
START = 0
POPULATION = 25
INDIVIDUAL = GRAPH - 1
GENERATIONS = 1000
PRINT_INTERVAL = 100
TESTS = 1
class Graph:
    def __init__(self, size=GRAPH, weight_range=WEIGHT_RANGE):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.nodes = list(range(size))
        self.fill_weights(weight_range)
        
    def __str__(self):
        return f'{self.nodes}'
    
    def get_matrix(self):
        return self.matrix
    
    def fill_weights(self, weight_range):
        n = self.size
        for i in range(n):
            for j in range(n):
                if (i != j):
                    self.matrix[i][j] = randint(*weight_range)
    
    def get_node_neighbors(self, node):
        if node not in self.nodes:
            print(f'{node} nao esta no grafo')
            return
    
        node_index = self.nodes.index(node)
        neighbors = []
        for i, weight in enumerate(self.matrix[node_index]):
            if weight != 0:
                neighbors.append(self.nodes[i])
        return neighbors
    
    def get_weight(self, n1, n2):
        index_1 = self.nodes.index(n1)
        index_2 = self.nodes.index(n2)
        
        return self.matrix[index_1][index_2]
                    
class TSP:
    def __init__(self, graph, start_n=START, p_size=POPULATION):
        self.graph = graph
        self.p_size = p_size
        self.start_node = start_n
        self.generations = []
        self.history = []
        self.greedy_cost = None

    def get_random_individual(self):
        labels = list(self.graph.nodes)
        labels.remove(self.start_node)
        shuffle(labels)
        individual = labels
        return individual
            
    def get_initial_population(self):
        pop = []
        greed = TSPGreedy(self.graph)
        greedy_path = greed.solve(START)
        pop.append(greedy_path)
        while (len(pop) < self.p_size):
            individual = self.get_random_individual()
            if (individual not in pop):
                pop.append(individual)
        return pop

    def print_generations(self, interval=PRINT_INTERVAL):
        for i in range(len(self.generations)):
            if (i % interval == 0):
                print(f'gen {i} best fit = {min(self.get_fitness(ind) for ind in self.generations[i])}')
    
    def get_fitness(self, individual):
        fitness_score = 0
        path = [self.start_node] + individual + [self.start_node]
        
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i+1]
            fitness_score += self.graph.get_weight(current_node, next_node)
        return fitness_score
    
    def get_fitness_sum(self, population):
        return sum(self.get_fitness(ind) for ind in population)
    
    def get_selected_parent(self, population):
        # optmizing selection by storing just once the list of fitnes
        fitness_list = [self.get_fitness(ind) for ind in population]
        worst_fit = max(fitness_list) + 1
        
        aptitude_list = [worst_fit - fit for fit in fitness_list]
        total_aptitude = sum(aptitude_list)
            
        spin = uniform(0, total_aptitude)
        current_sum = 0
        
        for i, individual in enumerate(population):
            current_sum += aptitude_list[i]
            
            if current_sum >= spin:
                return individual
        
        return choice(population)
    
    def get_crossover(self, p1, p2):
        length = len(p1)
        cross_point = randint(1, length - 1)
        
        child1 = p1[:cross_point] + [char for char in p2 if char not in p1[:cross_point]]
        child2 = p2[:cross_point] + [char for char in p1 if char not in p2[:cross_point]]
        
        return child1, child2

    def get_select_and_crossover(self, population):
        new_gen = []
        n = len(population)
        while (len(new_gen) < n):
            s1 = self.get_selected_parent(population)
            s2 = self.get_selected_parent(population)
            if (s1 != s2):
                c1, c2 = self.get_crossover(s1, s2)
                new_gen.append(c1)
                if (len(new_gen) != n):
                    new_gen.append(c2)
        return new_gen
        
    def mutation(self, individual):
        if random() >= MUTATION:
            return individual
        
        m1 = randint(0, len(individual) - 1)
        m2 = randint(0, len(individual) - 1)
        while m2 == m1:
            m1 = randint(0, len(individual) - 1)
        
        trade = list(individual)
        trade[m1], trade[m2] = trade[m2], trade[m1]
        return trade

    def put_new_generation(self):
        
        current_pop = self.generations[-1]
        elite_individual = min(current_pop, key=self.get_fitness)
        new_gen = self.get_select_and_crossover(current_pop)

        for i in range(len(new_gen)):
            new_gen[i] = self.mutation(new_gen[i])
        
        
        worst_new_individual = max(new_gen, key=self.get_fitness)
        worst_index = new_gen.index(worst_new_individual)
        
        new_gen[worst_index] = elite_individual
        self.generations.append(new_gen)


 
    def plot_performance(self):
        greedy_performance = [self.greedy_cost] * (GENERATIONS + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(self.history)), self.history, marker='o', linestyle='-', label='Algoritmo Genético', color='blue')
        plt.plot(range(len(greedy_performance)), greedy_performance, linestyle='--', color='red', label='Algoritmo Guloso')
        plt.title('Comparação de Desempenho: AG vs. Guloso')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness (Distância Total)')
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.waitforbuttonpress()
        plt.close()

    def execute(self):
        greedy_solver = TSPGreedy(self.graph)
        greedy_path = greedy_solver.solve(self.start_node)
        self.greedy_cost = greedy_solver.get_path_cost(greedy_path)
        
        self.generations.append(self.get_initial_population())
        self.history.append(min(self.get_fitness(ind) for ind in self.generations[-1]))
        for _ in range(GENERATIONS - 1):
            self.put_new_generation()
            self.history.append(min(self.get_fitness(ind) for ind in self.generations[-1]))
        
        self.plot_performance()
        
class TSPGreedy:
    def __init__(self, graph):
        self.graph = graph

    def solve(self, start_node_label):
        path = [start_node_label]
        visited_nodes = {start_node_label}
        current_node_label = start_node_label
        
        while len(visited_nodes) < self.graph.size:
            min_weight = INF
            next_node_label = None

            for i in range(self.graph.size):
                neighbor_label = self.graph.nodes[i]
                current_node_index = self.graph.nodes.index(current_node_label)
                if neighbor_label not in visited_nodes and self.graph.get_weight(current_node_index, i) < min_weight:
                    min_weight = self.graph.get_weight(current_node_index, i)
                    next_node_label = neighbor_label

            if next_node_label:
                path.append(next_node_label)
                visited_nodes.add(next_node_label)
                current_node_label = next_node_label
            else:
                break
        
        path.append(start_node_label)
        
        return path

    def get_path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            n1_label = path[i]
            n2_label = path[i+1]
            cost += self.graph.get_weight(n1_label, n2_label)
        return cost
        
def main():
    G = Graph(GRAPH)
    for i in range(0, TESTS):
        travel_salesman_problem = TSP(G)
        travel_salesman_problem.execute()

if __name__ == '__main__':
    main()