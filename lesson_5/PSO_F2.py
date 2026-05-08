import random
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------------- Problema G6 ----------------

X1_MIN, X1_MAX = -10, 10
X2_MIN, X2_MAX = -10, 10

POP_SIZE = 5
GENERATIONS = 20

W = 0.2
C1 = 1.7
C2 = 1.7

VMAX_X1 = 0.2 * (X1_MAX - X1_MIN)
VMAX_X2 = 0.2 * (X2_MAX - X2_MIN)

random.seed(42)


@dataclass
class Particula:
    pos: list
    vel: list
    pbest: list


def funcao_objetivo(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2


def violacao_total(x1, x2):
    #não tem restrições, então sempre retorna 0
    return 0


def eh_viavel(x1, x2):
    return violacao_total(x1, x2) == 0


def eh_melhor(pos_a, pos_b):
    """
    Retorna True se pos_a é melhor que pos_b.
    Critério:
    1. Viável vence inviável.
    2. Se ambas viáveis, vence menor f(x).
    3. Se ambas inviáveis, vence menor violação.
    """
    x1a, x2a = pos_a
    x1b, x2b = pos_b

    va = violacao_total(x1a, x2a)
    vb = violacao_total(x1b, x2b)

    if va == 0 and vb > 0:
        return True

    if va > 0 and vb == 0:
        return False

    if va == 0 and vb == 0:
        return funcao_objetivo(x1a, x2a) < funcao_objetivo(x1b, x2b)

    return va < vb


def limitar_posicao(pos):
    pos[0] = max(X1_MIN, min(X1_MAX, pos[0]))
    pos[1] = max(X2_MIN, min(X2_MAX, pos[1]))


def limitar_velocidade(vel):
    vel[0] = max(-VMAX_X1, min(VMAX_X1, vel[0]))
    vel[1] = max(-VMAX_X2, min(VMAX_X2, vel[1]))


def gerar_posicao_viavel():
    """
    Gera uma posição viável por rejeição.
    O G6 tem região viável estreita, então não é bom deixar a população nascer aleatória demais.
    """
    while True:
        x1 = random.uniform(X1_MIN, 20.0)
        x2 = random.uniform(X2_MIN, 10.0)

        if eh_viavel(x1, x2):
            return [x1, x2]


def criar_particula():
    pos = gerar_posicao_viavel()

    vel = [
        random.uniform(-1.0, 1.0),
        random.uniform(-1.0, 1.0)
    ]

    return Particula(
        pos=pos.copy(),
        vel=vel,
        pbest=pos.copy()
    )


def criar_populacao():
    return [criar_particula() for _ in range(POP_SIZE)]


def atualizar_particula(particula, gbest):
    for i in range(2):
        r1 = random.random()
        r2 = random.random()

        particula.vel[i] = (
            W * particula.vel[i]
            + C1 * r1 * (particula.pbest[i] - particula.pos[i])
            + C2 * r2 * (gbest[i] - particula.pos[i])
        )

    limitar_velocidade(particula.vel)

    nova_pos = [
        particula.pos[0] + particula.vel[0],
        particula.pos[1] + particula.vel[1]
    ]

    limitar_posicao(nova_pos)

    # Só aceita movimento se ele não piorar muito a viabilidade.
    # Isso evita colapso em (13, 0).
    if eh_melhor(nova_pos, particula.pos) or random.random() < 0.05:
        particula.pos = nova_pos

    if eh_melhor(particula.pos, particula.pbest):
        particula.pbest = particula.pos.copy()


def executar_pso():
    populacao = criar_populacao()

    gbest = populacao[0].pbest.copy()

    for p in populacao:
        if eh_melhor(p.pbest, gbest):
            gbest = p.pbest.copy()

    historico_f = []
    historico_violacao = []

    for geracao in range(GENERATIONS):
        for p in populacao:
            atualizar_particula(p, gbest)

            if eh_melhor(p.pbest, gbest):
                gbest = p.pbest.copy()

        f_gbest = funcao_objetivo(gbest[0], gbest[1])
        v_gbest = violacao_total(gbest[0], gbest[1])

        historico_f.append(f_gbest)
        historico_violacao.append(v_gbest)

        if geracao % 25 == 0:
            print(
                f"Geração {geracao:03d} | "
                f"x1 = {gbest[0]:.10f} | "
                f"x2 = {gbest[1]:.10f} | "
                f"f(x) = {f_gbest:.10f} | "
                f"violação = {v_gbest:.10e}"
            )

    return gbest, historico_f, historico_violacao, populacao


gbest, historico_f, historico_violacao, populacao = executar_pso()

x1, x2 = gbest

print("\n--- Melhor solução encontrada ---")
print(f"x1 = {x1:.10f}")
print(f"x2 = {x2:.10f}")
print(f"f(x) = {funcao_objetivo(x1, x2):.10f}")
print(f"violação total = {violacao_total(x1, x2):.10e}")

print("\n--- Solução conhecida ---")
print("x1 ≈ 1")
print("x2 ≈ 3")
print("f(x) ≈ 0")


plt.figure(figsize=(10, 6))
plt.plot(historico_f)
plt.xlabel("Geração")
plt.ylabel("Melhor f(x)")
plt.title("Convergência do PSO - G6")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergencia_pso_g6.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(historico_violacao)
plt.xlabel("Geração")
plt.ylabel("Violação total do melhor global")
plt.title("Violação das Restrições - PSO G6")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("violacao_pso_g6.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(
    [p.pos[0] for p in populacao],
    [p.pos[1] for p in populacao],
    alpha=0.7,
    label="Partículas finais"
)
plt.scatter([gbest[0]], [gbest[1]], marker="x", s=120, label="GBEST")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Distribuição Final das Partículas - G6")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("particulas_finais_pso_g6.png", dpi=300)
plt.show()