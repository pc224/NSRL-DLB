# config.py
# 全局参数配置

POP_SIZE = 60             # 种群大小 (论文中为100，此处为演示效率稍作调整)
MAX_GEN = 40              # 每个环境下的最大进化代数 (论文中为50)
NUM_WORKSTATIONS = 4      # 工作站数量
MAX_ROBOTS = 12           # 可用机器人总数
NUM_ENVIRONMENTS = 6      # 动态环境总数 (l = 0, 1, ...)
CROSSOVER_PROB = 0.6      # 交叉概率
MUTATION_PROB = 0.4       # 变异概率

# 因果推断参数
K_RATIO = 0.6             # 保留的因果特征比例 (论文中k=60%表现最佳)
ALPHA_CI = 0.05           # 条件独立检验的显著性水平 (p-value 阈值)
SVR_WINDOW = 3            # SVR预测的时间窗口大小 (q)