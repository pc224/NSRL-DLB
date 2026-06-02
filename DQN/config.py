class Config:
    TRAIN_STEPS = 50000
    BATCH_SIZE = 128
    LR = 0.001
    GAMMA = 0.99
    MEMORY_SIZE = 10000
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    TARGET_UPDATE_FREQ = 20


    N_TASKS = 10

    CYCLE_TIME = 40

    # ---------------- 训练设置 ----------------
    EPISODES = 1000
    DATASET_SIZE = 1000
    TEST_CASE_SEED = 42