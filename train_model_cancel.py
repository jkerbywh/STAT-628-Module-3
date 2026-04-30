import pandas as pd
import xgboost as xgb
import numpy as np
import gc
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

data_path = r"D:\文案\UWM\628 Data Science Practicum\module3\data\merged_airline_data_optimized.parquet"

print("1. 正在读取全量数据...")
df = pd.read_parquet(data_path)

# ⚠️ 关键改变：这里只清理 Cancelled 本身为空的极少数异常数据
# 绝对不能清理 ArrDelay 的空值，因为那正是被取消的航班！
df = df.dropna(subset=['Cancelled'])

# 检查一下数据中的取消率（通常在 1% - 3% 之间，属于极度不平衡数据）
cancel_rate = df['Cancelled'].mean() * 100
print(f"当前数据集中整体航班取消率为: {cancel_rate:.2f}%")

features = [
    'Month', 'DayofMonth', 'DayOfWeek', 'Distance', 'CRSDepTime', 'CRSArrTime',
    'Origin', 'Dest', 'Reporting_Airline'
]

print("2. 使用内存安全方式划分训练集(80%)和验证集(20%)...")
np.random.seed(42)
mask = np.random.rand(len(df)) < 0.8

X_train = df.loc[mask, features]
# 分类任务的标签必须是整数型 (0 或 1)
y_train = df.loc[mask, 'Cancelled'].astype('int32') 

X_val = df.loc[~mask, features]
y_val = df.loc[~mask, 'Cancelled'].astype('int32')

print("3. 正在释放原始数据表的内存...")
del df
del mask
gc.collect() 

print("4. 训练全量 XGBoost 分类模型 (引入早停机制)...")
# 注意：这里换成了 XGBClassifier (分类器)
model = xgb.XGBClassifier(
    n_estimators=300,        # 给足够的树让它去学
    max_depth=7,
    learning_rate=0.1,
    tree_method='hist',
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=15 # 如果验证集误差连续 15 棵树没有降低，就提前停止！
)

# 将验证集喂给模型用于实时监控
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=20  # 每 20 棵树打印一次验证成绩
)
print("✅ 模型训练完毕！")

print("\n" + "=" * 40)
print("5. 验证集最终评估结果:")

# Web App 真正需要的是“概率值”，而不是单纯的 0 或 1
# predict_proba 会返回每条数据为 0 和为 1 的概率，我们提取为 1 (取消) 的概率
y_pred_proba = model.predict_proba(X_val)[:, 1]

# 评估指标 1: ROC-AUC (衡量模型区分正常与取消航班的综合能力，0.5是瞎猜，越接近1越好)
auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"ROC-AUC 得分: {auc_score:.4f}")

# 评估指标 2: PR-AUC (由于取消的航班很少，这个指标比 ROC-AUC 更能反映真实水平)
pr_auc_score = average_precision_score(y_val, y_pred_proba)
print(f"PR-AUC (平均精度): {pr_auc_score:.4f}")
print("=" * 40)

