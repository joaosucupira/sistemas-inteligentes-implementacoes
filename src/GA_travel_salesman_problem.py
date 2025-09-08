import matplotlib.pyplot as plt
from sys import maxsize
from random import randint, random, choice, shuffle, uniform
from string import ascii_uppercase

INF = maxsize
GRAPH = 20
WEIGHT_RANGE = (1,10)
MUTATION = 0.02
START = 'A'
POPULATION = 100
INDIVIDUAL = GRAPH - 1
GENERATIONS = 500
PRINT_INTERVAL = 100

class Graph:
    def __init__(self, size=GRAPH, weight_range=WEIGHT_RANGE):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.nodes = list(ascii_uppercase[:size])
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
        labels = self.graph.get_node_neighbors(self.start_node)
        shuffle(labels)
        individual = "".join(labels)
        return individual
            
    def get_initial_population(self):
        pop = []
        while (len(pop) < self.p_size):
            individual = self.get_random_individual()
            if (individual not in pop):
                pop.append(individual)
        return pop

    def print_generations(self, interval=PRINT_INTERVAL):
        for i in range(0, len(self.generations)):
            if (i % interval == 0):
                print(f'gen {i} best fit = {min(self.get_fitness(ind) for ind in self.generations[i])}')
    
    def get_fitness(self, individual):
        # TODO: Calculate fitness function from path string
        fitness_score = 0
        start = self.start_node
        G = self.graph
        path = start + individual + start
        
        for i in range(len(path)- 1):
            current_node = path[i]
            next_node = path[i+1]
            
            fitness_score += G.get_weight(current_node, next_node)
        
        return fitness_score
    
    def get_fitness_sum(self, population):
        return sum(self.get_fitness(individual) for individual in population)
    
    def get_selected_parent(self, population):
        
        # getting the worst fit from the population
        worst_fit = max(self.get_fitness(ind) for ind in population) + 1
        total_aptitude = sum(worst_fit - self.get_fitness(ind) for ind in population)
        
        # spinning the wheel for a random value under the total fitness sum up
        spin = uniform(0, total_aptitude)
        s = 0
        
        for individual in population:
            
            # accumulates the sum and compares it to the gap between the worst fit and the current genes fitness
            fitness = worst_fit - self.get_fitness(individual)
            s += fitness
            
            # choosing wheel gets most probably the individuals with LESSER total cost
            if s >= spin:
                return individual
            
        # rarely, it will choose none, so just pick a random one/ Fallback
        return choice(population)
    
    def get_crossover(self, p1, p2):
        length = len(p1)
        cross_point = randint(1, length - 1)
        
        # order crossover
        # we use order crossover so we dont repeat/lose nodes for the solution
        # we get part of a chromossome and filter the second part with the ones missing according to the order not the exact sub string
        child1 = p1[:cross_point] + "".join(char for char in p2 if char not in p1[:cross_point])
        child2 = p2[:cross_point] + "".join(char for char in p1 if char not in p2[:cross_point])
        
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
        return "".join(trade)

    def put_new_generation(self):
        new_gen = self.get_select_and_crossover(self.generations[-1])  
        for i in range(0, len(new_gen)):
            new_gen[i] = self.mutation(new_gen[i])
                
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
        plt.show()

    def execute(self):
        greedy_solver = TSPGreedy(self.graph)
        greedy_path = greedy_solver.solve(self.start_node)
        self.greedy_cost = greedy_solver.get_path_cost(greedy_path)
        
        self.generations.append(self.get_initial_population())
        self.history.append(min(self.get_fitness(ind) for ind in self.generations[-1]))
        for _ in range(0, GENERATIONS - 1):
            self.put_new_generation()
            self.history.append(min(self.get_fitness(ind) for ind in self.generations[-1]))
            

        # print("Generations evolution")
        # self.print_generations()
        self.plot_performance()

class TSPGreedy:
    def __init__(self, graph):
        self.graph = graph

    def solve(self, start_node_label):
        path = [start_node_label]
        visited_nodes = {start_node_label}
        current_node_label = start_node_label
        
        while len(visited_nodes) < self.graph.size:
            current_node_index = self.graph.nodes.index(current_node_label)
            min_weight = INF
            next_node_label = None

            for i, weight in enumerate(self.graph.matrix[current_node_index]):
                neighbor_label = self.graph.nodes[i]
                if neighbor_label not in visited_nodes and weight < min_weight:
                    min_weight = weight
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
    travel_salesman_problem = TSP(G)
    travel_salesman_problem.execute()

    
if __name__ == '__main__':
    main()
        