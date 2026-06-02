import numpy as np


class DisassemblyEnv:


    def __init__(self, n_tasks, cycle_time, precedence_matrix, task_times):
        self.n_tasks = n_tasks
        self.CT = cycle_time
        self.TP = precedence_matrix  # N x N 优先关系矩阵
        self.task_times = task_times  # 长度 N 的作业时间

        # 状态维度: s1(1) + s6(1) + s7(1) + s2(N) + s3(N*N) + s4(N) + s5(N)
        self.state_dim = 3 + 3 * self.n_tasks + self.n_tasks ** 2
        self.action_dim = self.n_tasks

        self.reset()

    def reset(self):
        self.current_ws = 1  # s6: 当前工作站索引
        self.remaining_time = self.CT  # s7: 当前工作站剩余时间
        self.assigned = np.zeros(self.n_tasks, dtype=int)  # s4: 分配状态
        self.order = np.zeros(self.n_tasks, dtype=int)  # s5: 拆卸顺序
        self.current_step = 0

        self.mask = self._update_mask()
        return self._get_state(), self.mask

    def _get_state(self):

        s1 = 1.0  # CT / CT
        s6 = self.current_ws / self.n_tasks
        s7 = self.remaining_time / self.CT
        s2 = self.task_times / self.CT
        s3 = self.TP.flatten()
        s4 = self.assigned.copy()
        s5 = self.order / self.n_tasks

        state = np.concatenate(([s1, s6, s7], s2, s3, s4, s5))
        return state.astype(np.float32)

    def _update_mask(self):

        mask = np.ones(self.n_tasks, dtype=int)
        for i in range(self.n_tasks):
            if self.assigned[i] == 1:
                mask[i] = 1
            else:
                # 检查所有前驱任务是否都已分配
                preds = np.where(self.TP[:, i] == 1)[0]
                if len(preds) == 0 or np.all(self.assigned[preds] == 1):
                    mask[i] = 0
                else:
                    mask[i] = 1
        return mask

    def step(self, action):
        assert self.mask[action] == 0, "Selected action is invalid (masked)!"

        t_i = self.task_times[action]
        self.current_step += 1

        # 更新 s4 和 s5
        self.assigned[action] = 1
        self.order[action] = self.current_step

        new_ws_opened = False
        t_sum = 0

        if t_i > self.remaining_time:
            # 开启新工作站
            new_ws_opened = True
            t_sum = self.CT - self.remaining_time  # 前一个工作站的总作业时间
            self.current_ws += 1
            self.remaining_time = self.CT - t_i
        else:
            # 不开启新工作站
            self.remaining_time -= t_i

        reward = 1.0
        if new_ws_opened:
            # 惩罚前一个工作站的空闲时间
            reward -= (self.CT - t_sum) ** 2 / (self.CT ** 2)

        is_last_task = (self.current_step == self.n_tasks)
        if is_last_task:
            # 最后一个任务完成，惩罚最后一个工作站的空闲时间
            last_ws_time = self.CT - self.remaining_time
            reward -= (self.CT - last_ws_time) ** 2 / (self.CT ** 2)
            done = True
        else:
            done = False

        self.mask = self._update_mask()
        next_state = self._get_state()

        # info 用于后续解码拆卸方案
        info = {
            'ws_index': self.current_ws,
            'new_ws_opened': new_ws_opened
        }

        return next_state, reward, done, info, self.mask