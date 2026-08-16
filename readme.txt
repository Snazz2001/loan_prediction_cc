请帮我基于 data/loan_dataset_20000.csv,random_state为42，用sklearn的train_test_split来生成train，test数据，  构建一个标准信用评分卡模型。严格按照 CLAUDE.md 规范，调用 utils/risk_skills.py 里的技能完成特征工程、VIF过滤、逻辑回归/LightGBM建模，并在 test 上计算 KS、AUC 和 PSI。建模完成后，自动在 reports/ 下生成完整的 model_validation_report.md 文档。在scripts中生成train.py和test.py的代码来方便review和audit。

请帮我把 Optuna + LightGBM 调优脚本保存至 scripts/tune_lgb_skill.py。随后基于 data/loan_dataset_20000.csv 数据集，设置 random_state=42 并使用
  sklearn.model_selection.train_test_split 进行数据切分生成 train 和 test 数据集。在 train 数据集上调用 scripts/tune_lgb_skill.py 执行 50 轮 Optuna
  超参数寻优（5折交叉验证，以 AUC/KS 为目标），将最优模型在 test 数据集上进行测试评估（输出 AUC、KS 和混淆矩阵）。最后按照 CLAUDE.md 规范，将最佳超参数、特征重要性及 test
  数据集上的详细验证指标输出到 reports/model_validation_report.md 文档中。