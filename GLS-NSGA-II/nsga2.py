import random
import copy
from typing import List
from chromosome import Chromosome, decode
from operators import generate_individual, crossover, mutate, greedy_local_search
from problem_instance import Instance
from taog_model import Alternative


def fast_non_dominated_sort(pop: List[Chromosome]):
    """NSGA-II 快速非支配排序"""
    S = [[] for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    fronts = [[]]

    for p in range(len(pop)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop)):
            if (pop[p].f1 < pop[q].f1 and pop[p].f2 <= pop[q].f2) or \
                    (pop[p].f1 <= pop[q].f1 and pop[p].f2 < pop[q].f2):
                S[p].append(q)
            elif (pop[q].f1 < pop[p].f1 and pop[q].f2 <= pop[p].f2) or \
                    (pop[q].f1 <= pop[p].f1 and pop[q].f2 < pop[p].f2):
                n[p] += 1

        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts


def crowding_distance(pop: List[Chromosome], front: List[int]):
    """计算拥挤度距离"""
    if not front: return
    for i in front:
        pop[i].crowding_dist = 0

    for obj in ['f1', 'f2']:
        front.sort(key=lambda x: getattr(pop[x], obj))
        pop[front[0]].crowding_dist = float('inf')
        pop[front[-1]].crowding_dist = float('inf')

        f_min = getattr(pop[front[0]], obj)
        f_max = getattr(pop[front[-1]], obj)
        f_range = f_max - f_min if f_max > f_min else 1.0

        for i in range(1, len(front) - 1):
            pop[front[i]].crowding_dist += (getattr(pop[front[i + 1]], obj) - getattr(pop[front[i - 1]], obj)) / f_range


def run_nsga2(alt: Alternative, inst: Instance, Npop=50, Gen=100, ps=0.3) -> List[Chromosome]:
    """运行单条路径的 GLS-NSGA-II"""
    pop = [generate_individual(alt, inst) for _ in range(Npop)]

    for gen in range(Gen):
        fronts = fast_non_dominated_sort(pop)
        for f in fronts:
            crowding_distance(pop, f)

        # 锦标赛选择与生成子代
        offspring = []
        while len(offspring) < Npop:
            p1 = random.choice(pop)
            p2 = random.choice(pop)
            if p1.rank < p2.rank or (p1.rank == p2.rank and p1.crowding_dist > p2.crowding_dist):
                parent1 = p1
            else:
                parent1 = p2

            p3 = random.choice(pop)
            p4 = random.choice(pop)
            if p3.rank < p4.rank or (p3.rank == p4.rank and p3.crowding_dist > p4.crowding_dist):
                parent2 = p3
            else:
                parent2 = p4

            c1, c2 = crossover(parent1, parent2, alt)
            mutate(c1, alt, inst)
            mutate(c2, alt, inst)
            decode(c1, alt, inst)
            decode(c2, alt, inst)
            offspring.extend([c1, c2])

        # 合并种群并 elitism 选择
        combined = pop + offspring[:Npop]
        fronts = fast_non_dominated_sort(combined)
        for f in fronts:
            crowding_distance(combined, f)

        combined.sort(key=lambda x: (x.rank, -x.crowding_dist))
        pop = combined[:Npop]

        # 贪心局部搜索 (GLS)
        for ind in pop:
            if random.random() < ps:
                greedy_local_search(ind, alt, inst)

    # 返回 Pareto 前沿 (Rank 0)
    fronts = fast_non_dominated_sort(pop)
    return [pop[i] for i in fronts[0]]