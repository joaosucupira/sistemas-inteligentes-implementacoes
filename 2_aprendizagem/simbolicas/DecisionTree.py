import numpy as np
from collections import Counter


class Node:
    def __init__(self, atributo=None, valor_limite=None, ramo_esquerdo=None, ramo_direito=None,*,rotulo=None):
        self.atributo = atributo
        self.valor_limite = valor_limite
        self.ramo_esquerdo = ramo_esquerdo
        self.ramo_direito = ramo_direito
        self.rotulo = rotulo
        
    def e_folha(self):
        return self.rotulo is not None


class ArvoreDecisao:
    def __init__(self, amostras_minimas_divisao=2, profundidade_maxima=100, num_atributos=None):
        self.amostras_minimas_divisao = amostras_minimas_divisao
        self.profundidade_maxima = profundidade_maxima
        self.num_atributos = num_atributos
        self.raiz = None

    def fit(self, Dados, Alvos):
        self.num_atributos = Dados.shape[1] if not self.num_atributos else min(Dados.shape[1],self.num_atributos)
        self.raiz = self._crescer_arvore(Dados, Alvos)

    def _crescer_arvore(self, Dados, Alvos, nivel=0):
        num_amostras, num_caracteristicas = Dados.shape
        num_rotulos = len(np.unique(Alvos))

        if (nivel >= self.profundidade_maxima or num_rotulos == 1 or num_amostras < self.amostras_minimas_divisao):
            valor_folha = self._rotulo_mais_comum(Alvos)
            return Node(rotulo=valor_folha)

        indices_carac = np.random.choice(num_caracteristicas, self.num_atributos, replace=False)

        melhor_caracteristica, melhor_limite = self._melhor_divisao(Dados, Alvos, indices_carac)

        if melhor_caracteristica is None:
            valor_folha = self._rotulo_mais_comum(Alvos)
            return Node(rotulo=valor_folha)

        indices_esq, indices_dir = self._dividir(Dados[:, melhor_caracteristica], melhor_limite)
        esq = self._crescer_arvore(Dados[indices_esq, :], Alvos[indices_esq], nivel + 1)
        dir = self._crescer_arvore(Dados[indices_dir, :], Alvos[indices_dir], nivel + 1)
        return Node(melhor_caracteristica, melhor_limite, esq, dir)

    def _melhor_divisao(self, Dados, Alvos, indices_carac):
        melhor_ganho = -1
        indice_divisao, limite_divisao = None, None

        for indice_carac in indices_carac:
            Coluna_Dados = Dados[:, indice_carac]
            
            dados_validos = Coluna_Dados[~np.isnan(Coluna_Dados)]
            if len(dados_validos) < self.amostras_minimas_divisao:
                 continue
            
            limites = np.unique(dados_validos)
            
            for lim in limites:
                ganho = self._razao_ganho(Alvos, Coluna_Dados, lim)

                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    indice_divisao = indice_carac
                    limite_divisao = lim

        return indice_divisao, limite_divisao

    def _razao_ganho(self, Alvos, Coluna_Dados, limite):
        
        mascara_validos = ~np.isnan(Coluna_Dados)
        dados_validos = Alvos[mascara_validos]
        fracao_conhecida = len(dados_validos) / len(Alvos)
        
        if fracao_conhecida == 0:
            return 0
        
        entropia_pai = self._entropy(dados_validos)
        indices_esq, indices_dir = self._dividir(Coluna_Dados[mascara_validos], limite)
        
        Alvos_validos_esq = dados_validos[indices_esq]
        Alvos_validos_dir = dados_validos[indices_dir]

        if len(Alvos_validos_esq) == 0 or len(Alvos_validos_dir) == 0:
            return 0
        
        N_validos = len(dados_validos)
        N_esq, N_dir = len(Alvos_validos_esq), len(Alvos_validos_dir)
        E_esq, E_dir = self._entropy(Alvos_validos_esq), self._entropy(Alvos_validos_dir)
        entropia_filho_ponderada = (N_esq / N_validos) * E_esq + (N_dir / N_validos) * E_dir
        
        ganho_informacao = entropia_pai - entropia_filho_ponderada
        ganho_informacao_ajustado = ganho_informacao * fracao_conhecida 
        
        P_esq = N_esq / N_validos
        P_dir = N_dir / N_validos
        
        informacao_divisao = - (P_esq * np.log2(P_esq) + P_dir * np.log2(P_dir))  
        if informacao_divisao == 0:
            return 0
        
        razao_ganho = ganho_informacao_ajustado / informacao_divisao
        return razao_ganho

    def _dividir(self, Coluna_Dados, limite_divisao):
        indices_esq = np.argwhere(Coluna_Dados <= limite_divisao).flatten()
        indices_dir = np.argwhere(Coluna_Dados > limite_divisao).flatten()
        return indices_esq, indices_dir

    def _entropy(self, Alvos):
        contagem = np.bincount(Alvos)
        probabilidades = contagem / len(Alvos)
        return -np.sum([p * np.log(p) for p in probabilidades if p > 0])

    def _rotulo_mais_comum(self, Alvos):
        if len(Alvos) == 0:
            return None
        contagem = Counter(Alvos)
        valor = contagem.most_common(1)[0][0]
        return valor

    def predict(self, Dados):
        return np.array([self._atravessar_arvore(d, self.raiz) for d in Dados])

    def _atravessar_arvore(self, dado, no):
        if no.e_folha():
            return no.rotulo
        
        # Tratamento de Valores Ausentes (NaNs) durante a predição C4.5
        if np.isnan(dado[no.atributo]):
            # Em C4.5, distribui-se a predição fracionalmente (complexo), 
            # mas simplificamos para a predição majoritária do nó raiz do subárvore
            # Aqui, para manter a lógica simples, segue o caminho majoritário do treino.
            # Como a informação não está no Nó, retornamos a classe majoritária final (conservador)
            return self.raiz.rotulo 
            
        if dado[no.atributo] <= no.valor_limite:
            return self._atravessar_arvore(dado, no.ramo_esquerdo)
        return self._atravessar_arvore(dado, no.ramo_direito)
           
    def print_tree(self, nomes_caracteristicas, no=None, pre=""):
        if no is None:
            no = self.raiz 

        if no.e_folha():
            print(f"{pre}└── RÓTULO: {no.rotulo}")
            return

        indice_carac = no.atributo
        nome_carac = nomes_caracteristicas[indice_carac]
        
        
        print(f"{pre}┌── SE {nome_carac} <= {no.valor_limite:.4f}:")
        self.print_tree(nomes_caracteristicas, no.ramo_esquerdo, pre + "│    ")
        
        
        print(f"{pre}└── SE {nome_carac} > {no.valor_limite:.4f}:")
        self.print_tree(nomes_caracteristicas, no.ramo_direito, pre + "    ")
        
    def post_prune(self, Dados_Validacao, Alvos_Validacao):
        self._post_prune_recursive(self.raiz, Dados_Validacao, Alvos_Validacao)

    def _post_prune_recursive(self, no, Dados_V, Alvos_V):
        if len(Alvos_V) == 0:
            return

        if no.e_folha():
            return

        
        mascara_validos = ~np.isnan(Dados_V[:, no.atributo])
        Dados_V = Dados_V[mascara_validos]
        Alvos_V = Alvos_V[mascara_validos]
        
        if len(Alvos_V) < self.amostras_minimas_divisao:
            return

        
        mascara_esq = Dados_V[:, no.atributo] <= no.valor_limite
        mascara_dir = Dados_V[:, no.atributo] > no.valor_limite

        if no.ramo_esquerdo:
            self._post_prune_recursive(no.ramo_esquerdo, Dados_V[mascara_esq], Alvos_V[mascara_esq])
        if no.ramo_direito:
            self._post_prune_recursive(no.ramo_direito, Dados_V[mascara_dir], Alvos_V[mascara_dir])

        if no.ramo_esquerdo and no.ramo_direito and no.ramo_esquerdo.e_folha() and no.ramo_direito.e_folha():
            if len(Alvos_V) == 0:
                return

            rotulo_esq = no.ramo_esquerdo.rotulo
            rotulo_dir = no.ramo_direito.rotulo
            
            
            erros_esq = np.sum(Alvos_V[mascara_esq] != rotulo_esq)
            erros_dir = np.sum(Alvos_V[mascara_dir] != rotulo_dir)
            erros_sem_poda = erros_esq + erros_dir
            
            rotulo_folha = self._rotulo_mais_comum(Alvos_V)
            erros_com_poda = np.sum(Alvos_V != rotulo_folha)

            N = len(Alvos_V)
            
            
            e1 = (erros_sem_poda + 0.5) / N 
            e2 = (erros_com_poda + 0.5) / N

            if e2 <= e1:
                no.ramo_esquerdo = None
                no.ramo_direito = None
                no.rotulo = rotulo_folha