import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTree import ArvoreDecisao 
from RandomForests import RandomForest


# CONFIGURAÇÕES GLOBAIS
N_ITERACOES = 5  # Número de vezes que cada cenário será executado para média
TAMANHO_TESTE = 0.2
RANDOM_SEED = 42 # Semente base para reprodutibilidade

# Definição dos Cenários (MAPA DE HIPERPARÂMETROS)
cenarios = {
    "1. LINHA DE BASE (Mod.)": {
        "max_depth": 10, "split_min": 5, "n_arvores": 10
    },
    "2. ALTA VARIÂNCIA (Complexo)": {
        "max_depth": 25, "split_min": 2, "n_arvores": 25
    },
    "3. ALTO BIAS (Simples)": {
        "max_depth": 5, "split_min": 20, "n_arvores": 15
    }
}

df = pd.read_csv('../data/sinais_com_label.csv')

# Remocao de i, si1, si2 e g1 (As 3 primeiras e a penúltima)
for i in range(0, 3):
    df = df.drop(df.columns[0], axis=1)
    
df = df.drop(df.columns[-2], axis=1)

X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 

# Função auxiliar de acurácia
def calcular_acuracia(y_verdadeiro, y_predito):
    return np.sum(y_verdadeiro == y_predito) / len(y_verdadeiro)


# EXECUÇÃO
resultados_finais = {}

for nome_cenario, params in cenarios.items():
    
    # Listas para coletar acurácias de todas as iterações do cenário
    acuracias_c45 = []
    acuracias_random = []
    
    max_d = params['max_depth']
    min_s = params['split_min']
    n_a = params['n_arvores']
    
    print(f"\n--- EXECUTANDO CENÁRIO: {nome_cenario} ---")
    print(f"Parâmetros: D={max_d}, S={min_s}, Árvores={n_a}")

    for i in range(N_ITERACOES):
        # 1. Divisão Aleatória com Semente Única (para garantir que cada iteração seja diferente)
        semente_iteracao = RANDOM_SEED + i 
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TAMANHO_TESTE, random_state=semente_iteracao
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=semente_iteracao
        )
        
        # 2. Criação e Treinamento dos Modelos
        
        # C4.5 (Decision Tree)
        clf_c45 = ArvoreDecisao(profundidade_maxima=max_d, amostras_minimas_divisao=min_s)
        clf_c45.fit(X_train, y_train)
        clf_c45.post_prune(X_val, y_val)
        predictions_c45 = clf_c45.predict(X_test)
        acuracias_c45.append(calcular_acuracia(y_test, predictions_c45))

        # Random Forests
        clf_random = RandomForest(n_trees=n_a, max_depth=max_d, min_samples_split=min_s)
        clf_random.fit(X_train, y_train)
        predictions_random = clf_random.predict(X_test)
        acuracias_random.append(calcular_acuracia(y_test, predictions_random))


    # 3. Armazenamento e Impressão dos Resultados Agregados
    
    media_c45 = np.mean(acuracias_c45)
    std_c45 = np.std(acuracias_c45)
    
    media_random = np.mean(acuracias_random)
    std_random = np.std(acuracias_random)
    
    print(f"Média C4.5: {media_c45:.5f} (Std: {std_c45:.5f})")
    print(f"Média RF:   {media_random:.5f} (Std: {std_random:.5f})")

    resultados_finais[nome_cenario] = {
        "C45_Media": media_c45, "C45_Std": std_c45,
        "RF_Media": media_random, "RF_Std": std_random
    }

# Fim da execução