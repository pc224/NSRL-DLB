# encoding.py
import numpy as np
import networkx as nx
from utils import random_topological_sort
from config import NUM_WORKSTATIONS, MAX_ROBOTS


class Individual:
    def __init__(self, ts, tr, rw):
        self.TS = ts  # Task Sequence (拓扑排序)
        self.TR = tr  # Task-Robot assignment
        self.RW = rw  # Robot-Workstation assignment
        self.objectives = [float('inf'), float('inf')]
        self.rank = 0
        self.crowding_dist = 0.0


def initialize_population(problem, pop_size):
    """初始化种群"""
    pop = []
    for _ in range(pop_size):
        ts = random_topological_sort(problem.G)
        # 随机分配机器人和工作站
        tr = np.random.randint(0, MAX_ROBOTS, size=len(ts))
        rw = np.random.randint(0, NUM_WORKSTATIONS, size=MAX_ROBOTS)
        pop.append(Individual(ts, tr, rw))
    return pop


def evaluate(individual, problem):
    """
    解码并计算目标函数 F1 (Cycle Time), F2 (Robot Count)
    """
    times = problem.get_task_times()
    ws_times = np.zeros(NUM_WORKSTATIONS)
    used_robots = set()

    # 简化的解码逻辑：根据序列和分配计算各工作站负载
    for i, task in enumerate(individual.TS):
        robot = individual.TR[i]
        ws = individual.RW[robot]
        ws_times[ws] += times[task]
        used_robots.add(robot)

    # F1: 最小化最大工作站时间 (Cycle Time)
    # F2: 最小化使用的机器人数量
    individual.objectives = [np.max(ws_times), len(used_robots)]
    return individual


def solution_regeneration(individual, problem, executed_tasks, ongoing_task):
    """
    Algorithm 2: Solution Re-generation Method
    保留已执行和正在执行的任务分配，对剩余任务重新生成
    """
    # 1. 识别受影响的解部分
    fixed_tasks = set(executed_tasks)
    if ongoing_task is not None:
        fixed_tasks.add(ongoing_task)

    # 2. 提取固定部分的 TS, TR, RW
    fixed_ts = [t for t in individual.TS if t in fixed_tasks]
    fixed_tr_map = {individual.TS[i]: individual.TR[i] for i in range(len(individual.TS)) if
                    individual.TS[i] in fixed_tasks}

    # 3. 对未执行任务重新生成拓扑排序
    unexecuted_tasks = [t for t in individual.TS if t not in fixed_tasks]
    sub_G = problem.G.subgraph(unexecuted_tasks)
    new_unexecuted_ts = random_topological_sort(sub_G)

    # 合并序列
    new_ts = fixed_ts + new_unexecuted_ts

    # 4. 重新分配未执行任务的机器人
    new_tr = np.zeros(len(new_ts), dtype=int)
    for i, task in enumerate(new_ts):
        if task in fixed_tr_map:
            new_tr[i] = fixed_tr_map[task]
        else:
            new_tr[i] = np.random.randint(0, MAX_ROBOTS)

    # RW 保持不变或微调
    new_rw = individual.RW.copy()

    return Individual(new_ts, new_tr, new_rw)