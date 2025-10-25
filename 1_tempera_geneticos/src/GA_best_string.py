import random

# Constantes do AG
POPULATION_SIZE = 9
INDIVIDUAL_SIZE = 9
MUTATION_RATE = 0.05
GENERATIONS = 100
TESTES = 1000

def get_initial_population(population_size, individual_size):
    
    pop = []
    for _ in range(population_size):
        individual = ''.join(random.choice('01') for _ in range(individual_size))
        pop.append(individual)
    return pop

def fitness(individual):
    
    return individual.count('1')

def get_fitness_sum(population):
    
    return sum(fitness(ind) for ind in population)

def print_population(population):
    
    f_sum = get_fitness_sum(population)
    
    for individual in population:
        fit = fitness(individual)
        chances = int(round(fit / f_sum * 100, 2))
        print(f'{individual} - fitness = {fit} - chances = {chances}%')
    print(f'generation fitness = {f_sum}')

def get_parent(population):
    
    total_fitness = get_fitness_sum(population)
    
    spin = random.randint(0, total_fitness)
    current_sum = 0
    for individual in population:
        current_sum += fitness(individual)
        if current_sum >= spin:
            return individual
    
    return random.choice(population) # Fallback

def crossover(parent1, parent2):
    
    length = len(parent1)
    if length <= 1:
        return parent1, parent2

    cross_point = random.randint(1, length - 1)
    
    child1 = parent1[:cross_point] + parent2[cross_point:]
    child2 = parent2[:cross_point] + parent1[cross_point:]
    
    return child1, child2

def mutate(individual, rate):
    
    individual_list = list(individual)
    for i in range(len(individual_list)):
        if random.random() < rate:
            individual_list[i] = '1' if individual_list[i] == '0' else '0'
    return "".join(individual_list)

def update_population(population, mutation_rate):
    
    new_pop = []
    n = len(population)
    
    while len(new_pop) < n:
        parent1 = get_parent(population)
        parent2 = get_parent(population)
        
        # Garante que os pais são diferentes
        if parent1 != parent2:
            child1, child2 = crossover(parent1, parent2)
            
            mutated_child1 = mutate(child1, mutation_rate)
            mutated_child2 = mutate(child2, mutation_rate)
            
            new_pop.append(mutated_child1)
            
            if len(new_pop) < n:
                new_pop.append(mutated_child2)
    return new_pop

def ag_best_string(pop_size, ind_size, generations, mutation_rate):
    
    population = get_initial_population(pop_size, ind_size)
    # print("--- População Inicial ---")
    # print_population(population)
    
    for gen in range(1, generations + 1):
        # print(f'\n--- Geração {gen} ---')
        # population = update_population(population, mutation_rate)
        # print_population(population)
        
        best_fitness = max(fitness(individual) for individual in population)
        
        if best_fitness == ind_size:
            # print(f'\nSolução ótima encontrada na geração {gen}.')
            return gen
        elif gen == generations:
            # print(f'\nMelhor fitness final após {generations} gerações: {best_fitness}')
            return gen
        

    return -1

# Execução do programa
if __name__ == '__main__':
    # ag_best_string(POPULATION_SIZE, INDIVIDUAL_SIZE, GENERATIONS, MUTATION_RATE)
    # ag_best_string(POPULATION_SIZE, INDIVIDUAL_SIZE, GENERATIONS, MUTATION_RATE * 2)
    s = 0
    for i in range(0, TESTES):
        s += ag_best_string(POPULATION_SIZE, INDIVIDUAL_SIZE, GENERATIONS, MUTATION_RATE)
        
    media = s / TESTES
    print(f'media = {media}')
        
