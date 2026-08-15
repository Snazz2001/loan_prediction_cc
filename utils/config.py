"""
建模流水线共享配置：由 scripts/train.py 与 scripts/test.py 共同引用，
确保训练阶段与测试阶段使用完全一致的路径、随机种子与风控阈值。
"""

RANDOM_STATE = 42
DATA_PATH = "data/loan_dataset_20000.csv"
ARTIFACTS_DIR = "artifacts"
REPORT_PATH = "reports/model_validation_report.md"

RAW_TARGET_COL = "loan_paid_back"
TARGET = "default"  # 1 = 违约 (bad), 0 = 正常还款 (good)；由 1 - loan_paid_back 得到

TEST_SIZE = 0.3
WOE_BINS = 5

# 特征筛选
IV_MIN = 0.02
IV_SUSPICIOUS = 0.5  # 与 calculate_woe_iv 内 "Suspicious / Overfitting" 评级阈值一致
VIF_MAX = 5.0

# 判别能力
KS_MIN = 0.30
AUC_MIN = 0.70

# 稳定性 (PSI)
PSI_WATCH = 0.10
PSI_ACTION = 0.25

# LightGBM 超参数寻优 (scripts/tune_lgb_skill.py: tune_lightgbm_optuna)
TUNE_N_TRIALS = 50
TUNE_CV_FOLDS = 5
TUNE_METRIC = "auc"  # CV 目标函数使用 AUC；KS 作为 Train/Test 上的事后监控指标一并报告
