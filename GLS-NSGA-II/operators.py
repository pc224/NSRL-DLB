import random
import copy
from typing import List
from chromosome import Chromosome, decode
from problem_instance import Instance
from taog_model import Alternative


def make_valid_seq(raw_seq: List[str], alt: Alternative) -> List[str]:
    """
    修复序列，确保满足TAOG的优先级约束 (基于Kahn算法与优先级排序)
    """
    in_degree = {t: 0 for t in alt.tasks}
    adj = {t: [] for t in alt.tasks}
    for u, v in alt.edges:
        adj[u].append(v)
        in_degree[v] += 1

    priority = {t: i for i, t in enumerate(raw_seq)}
    queue = [t for t in alt.tasks if in_degree[t] == 0]
    valid_seq = []

    while queue:
        queue.sort(key=lambda x: priority.get(x, 999))
        u = queue.pop(0)
        valid_seq.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return valid_seq


def generate_individual(alt: Alternative, inst: Instance) -> Chromosome:
    """生成初始个体"""
    raw_seq = list(alt.tasks)
    random.shuffle(raw_seq)
    seq = make_valid_seq(raw_seq, alt)

    res_alloc = []
    for t in seq:
        t_type = inst.tasks[t]['type']
        if t_type == 'Complex':
            res_alloc.append(0)
        elif t_type == 'Hazardous':
            res_alloc.append(1)
        elif t_type == 'Interactive':
            res_alloc.append(2)
        else:
            res_alloc.append(random.choice([0, 1]))  # Normal task

    ind = Chromosome(seq, res_alloc)
    decode(ind, alt, inst)
    return ind


def crossover(p1: Chromosome, p2: Chromosome, alt: Alternative) -> tuple:
    """两点交叉 + 优先级修复"""
    size = len(p1.task_seq)
    pt1, pt2 = sorted(random.sample(range(size), 2))

    # 交叉片段
    o1_raw = p1.task_seq[:pt1] + p2.task_seq[pt1:pt2] + p1.task_seq[pt2:]
    o2_raw = p2.task_seq[:pt1] + p1.task_seq[pt1:pt2] + p2.task_seq[pt2:]

    o1_seq = make_valid_seq(o1_raw, alt)
    o2_seq = make_valid_seq(o2_raw, alt)

    # 资源分配均匀交叉
    o1_res = [p1.resource_alloc[i] if random.random() < 0.5 else p2.resource_alloc[i] for i in range(size)]
    o2_res = [p2.resource_alloc[i] if random.random() < 0.5 else p1.resource_alloc[i] for i in range(size)]

    return Chromosome(o1_seq, o1_res), Chromosome(o2_seq, o2_res)


def mutate(ind: Chromosome, alt: Alternative, inst: Instance):
    """变异操作"""
    # 1. 序列变异 (随机交换并修复)
    if random.random() < 0.5:
        i, j = random.sample(range(len(ind.task_seq)), 2)
        raw_seq = ind.task_seq[:]
        raw_seq[i], raw_seq[j] = raw_seq[j], raw_seq[i]
        ind.task_seq = make_valid_seq(raw_seq, alt)

    # 2. 资源变异 (仅针对 Normal 任务)
    for i, t in enumerate(ind.task_seq):
        if inst.tasks[t]['type'] == 'Normal' and random.random() < 0.1:
            ind.resource_alloc[i] = 1 - ind.resource_alloc[i]  # Flip 0<->1


def greedy_local_search(ind: Chromosome, alt: Alternative, inst: Instance):

    improved = True
    while improved:
        improved = False
        # 1. 序列局部搜索
        for i in range(len(ind.task_seq)):
            task = ind.task_seq[i]
            # 找到可行插入范围
            preds = [u for u, v in alt.edges if v == task]
            succs = [v for u, v in alt.edges if u == task]

            min_idx = max([ind.task_seq.index(p) for p in preds] + [-1]) + 1
            max_idx = min([ind.task_seq.index(s) for s in succs] + [len(ind.task_seq)]) - 1

            for j in range(min_idx, max_idx + 1):
                if i == j: continue
                new_seq = ind.task_seq[:]
                new_seq.pop(i)
                new_seq.insert(j, task)

                new_ind = Chromosome(new_seq, ind.resource_alloc[:])
                decode(new_ind, alt, inst)

                # 如果支配或单目标改进则接受
                if (new_ind.f1 <= ind.f1 and new_ind.f2 < ind.f2) or \
                        (new_ind.f1 < ind.f1 and new_ind.f2 <= ind.f2):
                    ind.task_seq = new_ind.task_seq
                    ind.f1, ind.f2 = new_ind.f1, new_ind.f2
                    improved = True
                    break
            if improved: break

        # 2. 资源局部搜索
        for i, t in enumerate(ind.task_seq):
            if inst.tasks[t]['type'] == 'Normal':
                new_res = ind.resource_alloc[:]
                new_res[i] = 1 - new_res[i]
                new_ind = Chromosome(ind.task_seq[:], new_res)
                decode(new_ind, alt, inst)

                if (new_ind.f1 <= ind.f1 and new_ind.f2 < ind.f2) or \
                        (new_ind.f1 < ind.f1 and new_ind.f2 <= ind.f2):
                    ind.resource_alloc = new_ind.resource_alloc
                    ind.f1, ind.f2 = new_ind.f1, new_ind.f2
                    improved = True
                    break