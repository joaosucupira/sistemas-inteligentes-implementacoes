import numpy as np
import random
import math

def criar_matriz_distancias(num_cidades):
    """Cria uma matriz de distâncias aleatórias para o TSP."""
    distancias = np.zeros((num_cidades, num_cidades))
    for i in range(num_cidades):
        for j in range(i + 1, num_cidades):
            dist = random.randint(10, 100) # Distância aleatória entre 10 e 100
            distancias[i][j] = dist
            distancias[j][i] = dist # A matriz é simétrica
    return distancias

# Exemplo de uso:
num_cidades = 5
distancias = criar_matriz_distancias(num_cidades)
print(distancias)

# função objetiva
def calcular_custo(rota, distancias):
    """Calcula o custo total de uma rota."""
    custo = 0
    for i in range(len(rota) - 1):
        custo += distancias[rota[i]][rota[i+1]]
    custo += distancias[rota[-1]][rota[0]] # Retorno à cidade de origem
    return custo

def vizinho_2_opt(rota):
    """Gera uma rota vizinha trocando 2 segmentos (2-opt)."""
    nova_rota = rota[:]
    i, j = sorted(random.sample(range(1, len(nova_rota)), 2))
    nova_rota[i:j] = nova_rota[i:j][::-1]
    return nova_rota

def simulated_annealing_tsp(distancias, T_inicial, alpha, T_final):
    num_cidades = len(distancias)
    rota_atual = list(range(num_cidades))
    random.shuffle(rota_atual) # Solução inicial aleatória
    
    melhor_rota = rota_atual[:]
    melhor_custo = calcular_custo(melhor_rota, distancias)
    
    T = T_inicial
    
    while T > T_final:
        rota_vizinha = vizinho_2_opt(rota_atual)
        custo_vizinho = calcular_custo(rota_vizinha, distancias)
        custo_atual = calcular_custo(rota_atual, distancias)
        
        delta_custo = custo_vizinho - custo_atual
        
        if delta_custo < 0 or random.random() < math.exp(-delta_custo / T):
            rota_atual = rota_vizinha
            
        if calcular_custo(rota_atual, distancias) < melhor_custo:
            melhor_rota = rota_atual[:]
            melhor_custo = calcular_custo(melhor_rota, distancias)
            
        T *= alpha # função de decaimento da temperatura
        
    return melhor_rota, melhor_custo

# --- Exemplo de uso ---
num_cidades = 20
distancias = criar_matriz_distancias(num_cidades)

T_inicial = 10000
alpha = 0.99
T_final = 1

melhor_rota_encontrada, custo_final = simulated_annealing_tsp(distancias, T_inicial, alpha, T_final)

print(f"Número de cidades: {num_cidades}")
print(f"Melhor rota encontrada: {melhor_rota_encontrada}")
print(f"Custo da melhor rota: {custo_final}")