# causal_inference.py
import numpy as np
from scipy import stats
from sklearn.svm import SVR
from config import K_RATIO, ALPHA_CI, SVR_WINDOW


def partial_correlation(x, y, z):
    """
    计算偏相关系数 (控制变量z下x和y的相关性)
    用于近似条件独立检验 (CI-test)
    """
    # 线性回归求残差
    slope_x, intercept_x, _, _, _ = stats.linregress(z, x)
    res_x = x - (slope_x * z + intercept_x)

    slope_y, intercept_y, _, _, _ = stats.linregress(z, y)
    res_y = y - (slope_y * z + intercept_y)

    # 残差的 Pearson 相关系数
    r, p_value = stats.pearsonr(res_x, res_y)
    return r, p_value


def causal_feature_selection(history_X, history_Y, seed_var_idx):
    """
    Algorithm 3 (Lines 1-9): 基于条件独立检验的因果特征筛选
    history_X: shape (num_envs, num_features) 历史环境特征 (如任务平均排序位置)
    history_Y: shape (num_envs,) 历史环境标签 (Pareto前沿质量评分)
    seed_var_idx: 种子变量索引 (状态改变的任务ID)
    """
    num_features = history_X.shape[1]
    p_values = []

    seed_data = history_X[:, seed_var_idx]

    for i in range(num_features):
        if i == seed_var_idx:
            p_values.append(0.0)  # 种子变量必定保留
            continue

        # CI-test: 检验 X_i 与 Y 在给定 C0 (seed) 下是否条件独立
        _, p_val = partial_correlation(history_X[:, i], history_Y, seed_data)
        p_values.append(p_val)

    # 排序并保留 Top-K 因果特征 (p-value 越小，越不独立，越可能是因果特征)
    p_values = np.array(p_values)
    # 处理 NaN
    p_values[np.isnan(p_values)] = 1.0

    sorted_indices = np.argsort(p_values)
    k = int(num_features * K_RATIO)
    causal_features = sorted_indices[:k]

    return causal_features, p_values


def predict_guided_population(history_X, causal_features, current_seed_val, pop_size):
    """
    Algorithm 3 (Lines 10-20): 基于 SVR 预测因果特征并生成引导种群
    """
    num_envs = history_X.shape[0]
    guided_weights = np.zeros(history_X.shape[1])

    # 对每个因果特征训练 SVR 并预测新环境的值
    for feat_idx in causal_features:
        y_train = history_X[:, feat_idx]
        # 构建时间序列特征 (滑动窗口)
        if num_envs <= SVR_WINDOW:
            # 数据不足，使用简单平均或线性外推
            guided_weights[feat_idx] = np.mean(y_train)
        else:
            X_train = []
            y_train_seq = []
            for i in range(num_envs - SVR_WINDOW):
                X_train.append(y_train[i: i + SVR_WINDOW])
                y_train_seq.append(y_train[i + SVR_WINDOW])

            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr.fit(X_train, y_train_seq)

            # 预测下一个环境
            last_window = y_train[-SVR_WINDOW:].reshape(1, -1)
            pred_val = svr.predict(last_window)[0]
            guided_weights[feat_idx] = np.clip(pred_val, 0, 1)

    # 非因果特征赋予随机权重
    non_causal = [i for i in range(history_X.shape[1]) if i not in causal_features]
    guided_weights[non_causal] = np.random.rand(len(non_causal))

    return guided_weights