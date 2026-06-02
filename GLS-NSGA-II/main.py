import time
from problem_instance import create_mock_instance
from nsga2 import run_nsga2, fast_non_dominated_sort
from chromosome import Chromosome


def main():
    print("=== 初始化问题实例 ===")
    inst = create_mock_instance()
    print(f"生成包含 {len(inst.tasks)} 个任务的 TAOG 模型。")

    print("\n=== Phase 1: 从 TAOG 提取所有可行的拆卸替代方案 (DFS) ===")
    alternatives = inst.taog.extract_all_alternatives()
    print(f"共提取出 {len(alternatives)} 条可行的拆卸路径 (Alternatives)。")

    print("\n=== Phase 2: 对每条路径运行 GLS-NSGA-II ===")
    global_pareto = []


    limit_alts = min(5, len(alternatives))

    for idx, alt in enumerate(alternatives[:limit_alts]):
        print(f"正在处理 Alternative {idx + 1}/{limit_alts} (包含 {len(alt.tasks)} 个任务)...")
        start_time = time.time()

        # 运行 NSGA-II (参数设置参考论文 Table 5)
        pareto_front = run_nsga2(alt, inst, Npop=40, Gen=50, ps=0.2)
        global_pareto.extend(pareto_front)

        print(f"  -> 耗时: {time.time() - start_time:.2f}s, 获得非支配解数量: {len(pareto_front)}")

    print("\n=== 全局 Pareto 前沿合并与排序 ===")
    # 对所有路径得到的解进行全局非支配排序
    fronts = fast_non_dominated_sort(global_pareto)
    final_pareto = [global_pareto[i] for i in fronts[0]]

    # 去重 (基于目标函数值)
    unique_pareto = []
    seen = set()
    for ind in final_pareto:
        sig = (round(ind.f1, 2), round(ind.f2, 2))
        if sig not in seen:
            seen.add(sig)
            unique_pareto.append(ind)

    print(f"最终全局 Pareto 前沿包含 {len(unique_pareto)} 个非支配解。")
    print("\n--- Top 5 代表性解 (按成本 f1 排序) ---")
    unique_pareto.sort(key=lambda x: x.f1)
    for i, ind in enumerate(unique_pareto[:5]):
        print(
            f"解 {i + 1}: 总成本(f1) = {ind.f1:.2f}, 负载平滑指数(f2) = {ind.f2:.2f}, 开启工作站数 = {max(ind.station_assign.values())}")


if __name__ == "__main__":
    main()