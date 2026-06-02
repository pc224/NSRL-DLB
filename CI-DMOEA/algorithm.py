# algorithm.py
import numpy as np
from utils import fast_non_dominated_sort, crowding_distance
from encoding import initialize_population, evaluate, solution_regeneration, Individual
from causal_inference import causal_feature_selection, predict_guided_population
from config import POP_SIZE, MAX_GEN, CROSSOVER_PROB, MUTATION_PROB, MAX_ROBOTS, NUM_WORKSTATIONS


def nsga2_evolution(pop, problem, max_gen):
    for gen in range(max_gen):
        for ind in pop:
            evaluate(ind, problem)

        fronts = fast_non_dominated_sort(pop)
        for i, front in enumerate(fronts):
            crowding_distance(pop, front)

        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)

            child_ts = crossover_ts(p1.TS, p2.TS, problem.G)
            child_tr = crossover_arr(p1.TR, p2.TR)
            child_rw = crossover_arr(p1.RW, p2.RW)

            if np.random.rand() < MUTATION_PROB:
                mutate(child_ts, child_tr, problem.G)

            offspring.append(Individual(child_ts, child_tr, child_rw))

        combined = pop + offspring
        for ind in combined: evaluate(ind, problem)
        fronts = fast_non_dominated_sort(combined)
        for i, front in enumerate(fronts):
            crowding_distance(combined, front)

        combined.sort(key=lambda x: (x.rank, -x.crowding_dist))
        pop = combined[:POP_SIZE]

    return pop


def tournament_select(pop):
    idx1, idx2 = np.random.choice(len(pop), 2, replace=False)
    return pop[idx1] if pop[idx1].rank < pop[idx2].rank else pop[idx2]


def crossover_ts(ts1, ts2, G):
    new_ts = []
    used = set()
    while len(new_ts) < len(ts1):
        available = [t for t in ts1 if t not in used and all(p in used for p in G.predecessors(t))]
        if not available:
            remaining = [t for t in ts1 if t not in used]
            if remaining:
                chosen = np.random.choice(remaining)
            else:
                break
        else:
            chosen = np.random.choice(available)
        new_ts.append(chosen)
        used.add(chosen)
    return new_ts


def crossover_arr(arr1, arr2):
    """数组均匀交叉"""
    min_len = min(len(arr1), len(arr2))
    mask = np.random.rand(min_len) < 0.5
    res = np.copy(arr1)
    res[:min_len] = np.where(mask, arr1[:min_len], arr2[:min_len])
    return res


def mutate(ts, tr, G):
    """变异操作"""
    if len(ts) > 2:
        i = np.random.randint(0, len(ts) - 1)
        t1, t2 = ts[i], ts[i + 1]
        if t1 not in G.predecessors(t2) and t2 not in G.predecessors(t1):
            ts[i], ts[i + 1] = ts[i + 1], ts[i]
    if len(tr) > 0:
        idx = np.random.randint(0, len(tr))
        tr[idx] = np.random.randint(0, MAX_ROBOTS)


class CI_DMOEA:
    def __init__(self, problem):
        self.problem = problem
        self.history_X = []
        self.history_Y = []
        self.history_POS = []

    def run(self, num_envs):
        print("Starting CI-DMOEA Optimization...")
        pop = initialize_population(self.problem, POP_SIZE)

        for l in range(num_envs):
            print(f"\n--- Environment {l} ---")

            if l > 0:
                changed_task, executed, ongoing = self.problem.trigger_dynamic_change()
                print(f"Environment changed! Task {changed_task} damaged.")

                new_pop = []
                for ind in pop:
                    new_ind = solution_regeneration(ind, self.problem, executed, ongoing)
                    new_pop.append(new_ind)
                pop = new_pop

            if l > 3:
                seed_var = changed_task if changed_task != -1 else 0
                causal_feats, _ = causal_feature_selection(
                    np.array(self.history_X), np.array(self.history_Y), seed_var
                )
                weights = predict_guided_population(
                    np.array(self.history_X), causal_feats, 1.0, POP_SIZE
                )
                guided_pop = self._generate_guided_pop(weights)
                pop = pop[:POP_SIZE // 2] + guided_pop[:POP_SIZE // 2]

            pop = nsga2_evolution(pop, self.problem, MAX_GEN)

            pos = [ind for ind in pop if ind.rank == 0]
            self.history_POS.append(pos)

            feat_x = self._extract_features(pos)
            self.history_X.append(feat_x)
            avg_f1 = np.mean([ind.objectives[0] for ind in pos]) if pos else 100.0
            self.history_Y.append(1.0 / (avg_f1 + 1e-5))

            print(f"Env {l} finished. POS size: {len(pos)}, Avg F1: {avg_f1:.2f}")

    def _extract_features(self, pos):
        """提取种群特征：每个任务在 Pareto 解集中的平均相对排序位置"""
        feats = np.zeros(self.problem.num_tasks)
        counts = np.zeros(self.problem.num_tasks)
        for ind in pos:
            for rank, task in enumerate(ind.TS):
                feats[task] += rank / len(ind.TS)
                counts[task] += 1
        counts[counts == 0] = 1
        return feats / counts

    def _generate_guided_pop(self, weights):
        """根据预测权重生成引导种群"""
        pop = []
        w_sum = np.sum(weights)
        probs = weights / w_sum if w_sum > 0 else np.ones(len(weights)) / len(weights)

        for _ in range(POP_SIZE):
            ts = self._weighted_topological_sort(probs)
            tr = np.random.randint(0, MAX_ROBOTS, size=len(ts))
            rw = np.random.randint(0, NUM_WORKSTATIONS, size=MAX_ROBOTS)
            pop.append(Individual(ts, tr, rw))
        return pop

    def _weighted_topological_sort(self, weights):
        """带权重的拓扑排序"""
        G = self.problem.G
        in_degree = {node: len(list(G.predecessors(node))) for node in G.nodes()}
        queue = [node for node in G.nodes() if in_degree[node] == 0]
        topo_sort = []

        while queue:
            w = np.array([weights[n] for n in queue])
            w_sum = np.sum(w)
            w = w / w_sum if w_sum > 0 else np.ones(len(queue)) / len(queue)
            idx = np.random.choice(len(queue), p=w)
            node = queue.pop(idx)
            topo_sort.append(node)
            for successor in G.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        return topo_sort