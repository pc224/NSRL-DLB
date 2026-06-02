# utils.py
import numpy as np
import networkx as nx


def fast_non_dominated_sort(population):
    """NSGA-II 非支配排序"""
    n = len(population)
    domination_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    front = [[]]

    for p in range(n):
        for q in range(n):
            if p == q: continue
            # 假设个体有 .objectives 属性 (F1, F2)
            if dominates(population[p].objectives, population[q].objectives):
                dominated_solutions[p].append(q)
            elif dominates(population[q].objectives, population[p].objectives):
                domination_count[p] += 1

        if domination_count[p] == 0:
            population[p].rank = 0
            front[0].append(p)

    i = 0
    while len(front[i]) > 0:
        next_front = []
        for p in front[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 1
                    next_front.append(q)
        i += 1
        front.append(next_front)

    return front[:-1]  # 移除最后一个空列表


def dominates(obj1, obj2):
    """判断 obj1 是否支配 obj2 (最小化问题)"""
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))


def crowding_distance(population, front):
    """计算拥挤度距离"""
    n = len(front)
    if n == 0: return
    distances = np.zeros(n)

    for m in range(len(population[front[0]].objectives)):
        sorted_indices = np.argsort([population[i].objectives[m] for i in front])
        distances[0] = float('inf')
        distances[-1] = float('inf')

        obj_min = population[front[sorted_indices[0]]].objectives[m]
        obj_max = population[front[sorted_indices[-1]]].objectives[m]

        if obj_max - obj_min == 0: continue

        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (population[front[sorted_indices[i + 1]]].objectives[m] -
                                             population[front[sorted_indices[i - 1]]].objectives[m]) / (
                                                        obj_max - obj_min)

    for i, idx in enumerate(front):
        population[idx].crowding_dist = distances[i]


def random_topological_sort(G):
    """随机生成满足优先关系的拓扑排序序列"""
    in_degree = {node: len(list(G.predecessors(node))) for node in G.nodes()}
    queue = [node for node in G.nodes() if in_degree[node] == 0]
    topo_sort = []

    while queue:
        np.random.shuffle(queue)
        node = queue.pop(0)
        topo_sort.append(node)
        for successor in G.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)
    return topo_sort