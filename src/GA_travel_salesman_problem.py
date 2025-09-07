import matplotlib
from random import randint, random, choice, shuffle, uniform
from string import ascii_uppercase

GRAPH = 4
WEIGHT_RANGE = (1,10)
MUTATION = 0.05
START = 'A'
POPULATION = 5
INDIVIDUAL = GRAPH - 1
GENERATIONS = 5

class Graph:
    def __init__(self, size=GRAPH, weight_range=WEIGHT_RANGE):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.nodes = list(ascii_uppercase[:size])
        self.fill_weights(weight_range)
        print(self.matrix)
        
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

    def print_generations(self):
        for i in range(0, len(self.generations)):
            print(f'gen {i}')
            for individual in self.generations[i]:
                print(f'{individual} - {self.get_fitness(individual)}')
            print(self.get_fitness_sum(self.generations[i]))        
    
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
    
    def get_select_parents(self, population):
        parents = []
        fit_sum = self.get_fitness_sum(population)
        worst_fit = max(self.get_fitness(ind) for ind in population) + 1
        while (len(parents) < len(population)):
            spin = uniform(0, fit_sum)
            s = 0
            
            for individual in population:
                fitness = worst_fit - self.get_fitness(individual)
                s += fitness
                
                # MAIOR IGUAL
                if s >= spin:
                    parents.append(individual)
                    break

        return parents
    
    def crossover(self, p1, p2):
        length = len(p1)
        cross_point = randint(1, length - 1)
        
        # order crossover
        child1 = p1[:cross_point] + "".join(char for char in p2 if char not in p1[:cross_point])
        child2 = p2[:cross_point] + "".join(char for char in p1 if char not in p2[:cross_point])
        
        return child1, child2

    def get_crossover(population):
        crossed = []
        
        while (len(crossed) < len(population)):
            pass
            
        
        return crossed
        
    
    def mutation(self, individual):
        pass

    def put_new_generation(self):
        new_gen = self.get_select_parents(self.generations[-1])
        new_gen = self.get_crossover(new_gen)
        
        self.generations.append(new_gen)

    def execute(self):
        self.generations.append(self.get_initial_population())
        for _ in range(0, GENERATIONS - 1):
            self.put_new_generation()
        self.print_generations()
        
    

        
def main():
    G = Graph(GRAPH)
    travel_salesman_problem = TSP(G)
    travel_salesman_problem.execute()

    
if __name__ == '__main__':
    main()
        