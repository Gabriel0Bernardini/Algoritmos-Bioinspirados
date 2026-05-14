# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético com representação real (sem limite de bits)
# -----------------------------------------------------------------------------------

#Não funciona perfeitamente, talvez em função da restrição de igualdade, que é difícil de ser satisfeita com mutações aleatórias.

import random
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
POP_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.1
MUTATION_STD = 0.10
ELITISM = 7
PENALTY_LAMBDA = 100.0

X1_MIN, X1_MAX = -1.0, 1.0
X2_MIN, X2_MAX = -1.0, 1.0

# ---------------- Funções do problema ----------------

def funcao_objetivo(x1, x2):
    return x1**2 + (x2 - 1)**2


def restricao(x1, x2):
    return x2 - x1**2

def fitness(individuo):
    x1, x2 = individuo
    f = funcao_objetivo(x1, x2)
    penalidade = PENALTY_LAMBDA * abs(restricao(x1, x2))
    return f + penalidade

# ---------------- Funções do AG ----------------

def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))

def criar_individuo():
    x1 = random.uniform(X1_MIN, X1_MAX)
    x2 = random.uniform(X2_MIN, X2_MAX)
    return [x1, x2]

def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def roleta(populacao):
    fit_vals = [1 / fitness(ind) for ind in populacao]
    
    soma_fit = sum(fit_vals)
    
    probabilidades = [f / soma_fit for f in fit_vals]
    
    cumulativas = []
    acumulado = 0
    for p in probabilidades:
        acumulado += p
        cumulativas.append(acumulado)
    
    valor_sorteado = random.uniform(0, 1)
    
    for i, cum in enumerate(cumulativas):
        if valor_sorteado <= cum:
            return populacao[i]
        
def crossover_blx_alpha(pai1, pai2):
        alpha = 0.5

        filho1 = []
        filho2 = []
        
        # print("\n")
        # print(f"Pai 1:{pai1}")
        # print(f"Pai 2:{pai2}")

        for i in range(len(pai1)):
            #print("\n")
            d = abs(pai1[i] - pai2[i])
            limite_inf = min(pai1[i],pai2[i]) - alpha * d
            limite_sup = max(pai1[i],pai2[i]) + alpha * d
            filho1.append(random.uniform(limite_inf, limite_sup))
            filho2.append(random.uniform(limite_inf, limite_sup))

            # print(f"Iteracao {i}: Distancia({d})")
            # print(f"LI = {limite_inf}, LS = {limite_sup}")
            # print(f"Filho 1: {filho1[i]}")
            # print(f"Filho 2: {filho2[i]}")   

        return filho1,filho2 

def mutacao(individuo):
    x1, x2 = individuo

    if random.random() < MUTATION_RATE:
        x1 += random.gauss(0, MUTATION_STD)

    if random.random() < MUTATION_RATE:
        x2 += random.gauss(0, MUTATION_STD)

    x1 = limitar(x1, X1_MIN, X1_MAX)
    x2 = limitar(x2, X2_MIN, X2_MAX)

    return [x1, x2]

def nova_geracao(populacao):
    populacao_ordenada = sorted(populacao, key=fitness)

    nova_pop = [ind[:] for ind in populacao_ordenada[:ELITISM]]

    while len(nova_pop) < POP_SIZE:
        pai1 = roleta(populacao)
        pai2 = roleta(populacao)

        filho1, filho2 = crossover_blx_alpha(pai1, pai2)

        filho1 = mutacao(filho1)
        filho2 = mutacao(filho2)

        nova_pop.append(filho1)
        if len(nova_pop) < POP_SIZE:
            nova_pop.append(filho2)

    return nova_pop

# ---------------- Execução ----------------

populacao = criar_populacao()

melhores_fitness = []
melhores_f_obj = []
melhores_violacoes = []

#print("População inicial:")
# for i, individuo in enumerate(populacao):
#     x1, x2 = individuo
#     print(
#         f"Indivíduo {i}: "
#         f"x1 = {x1:.6f}, "
#         f"x2 = {x2:.6f}, "
#         f"f(x) = {funcao_objetivo(x1, x2):.6f}, "
#         f"violação = {abs(restricao(x1, x2)):.6f}, "
#         f"fitness = {fitness(individuo):.6f}"
#     )

for geracao in range(GENERATIONS):
    melhor = min(populacao, key=fitness)
    x1_best, x2_best = melhor

    best_f = funcao_objetivo(x1_best, x2_best)
    best_violation = abs(restricao(x1_best, x2_best))
    best_fit = fitness(melhor)

    melhores_fitness.append(best_fit)
    melhores_f_obj.append(best_f)
    melhores_violacoes.append(best_violation)

    if geracao % 10 == 0:
        print(
            f"Geração {geracao}: "
            f"x1 = {x1_best:.6f}, "
            f"x2 = {x2_best:.6f}, "
            f"f(x) = {best_f:.6f}, "
            f"violação = {best_violation:.6f}, "
            f"fitness = {best_fit:.6f}"
        )

    populacao = nova_geracao(populacao)

# ---------------- Resultado final ----------------

melhor_final = min(populacao, key=fitness)
x1_final, x2_final = melhor_final
f_final = funcao_objetivo(x1_final, x2_final)
violacao_final = abs(restricao(x1_final, x2_final))
fitness_final = fitness(melhor_final)

print("\n--- Melhor solução encontrada ---")
print(f"x1 = {x1_final:.10f}")
print(f"x2 = {x2_final:.10f}")
print(f"f(x) = {f_final:.10f}")
print(f"Violação da restrição = {violacao_final:.10f}")
print(f"Fitness penalizado = {fitness_final:.10f}")

print("\n--- Ótimo teórico esperado ---")
print("x1 = ±0.7071067812")
print("x2 = 0.5000000000")
print("f(x) = 0.7500000000")

# ---------------- Gráfico 1: função objetivo ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_f_obj, linewidth=2, label="f(x) (melhor)")
plt.axhline(y=0.75, linestyle='--', label='Ótimo (0.75)')

plt.xlabel("Geração")
plt.ylabel("f(x)")
plt.title("Convergência da função objetivo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_funcao_objetivo.png", dpi=300)
plt.close()


# ---------------- Gráfico 2: fitness ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_fitness, linewidth=2, label="Fitness penalizado")

plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência do fitness")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_fitness.png", dpi=300)
plt.close()


# ---------------- Gráfico 3: restrição ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_violacoes, linewidth=2, label="Violação da restrição")

plt.xlabel("Geração")
plt.ylabel("|x2 - x1²|")
plt.title("Convergência da restrição")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_restricao.png", dpi=300)
plt.close()

print("\nGráficos salvos com sucesso:")
print(" - grafico_funcao_objetivo.png")
print(" - grafico_fitness.png")
print(" - grafico_restricao.png")