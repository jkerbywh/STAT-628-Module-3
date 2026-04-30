import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. 读取数据
data_path = r"D:\文案\UWM\628 Data Science Practicum\module3\data\merged_airline_data_optimized.parquet"
print("1. 读取数据...")
df = pd.read_parquet(data_path)
df = df.dropna(subset=['ArrDelay'])

features = [
    'Month', 'DayofMonth', 'DayOfWeek', 'Distance', 'CRSDepTime', 'CRSArrTime',
    'Origin', 'Dest', 'Reporting_Airline'
]

# 为了快速跑通这个更复杂的模型，我们强烈建议先用 5% 数据抽样测试
# 等代码无 bug 后，再用全量数据跑！
print("2. 抽样 5% 进行训练 (后续可改为全量)...")
df_sample = df.sample(frac=0.05, random_state=42)

# 处理类别特征（为 sklearn 准备，直接转为数值编码）
# 虽然没有 XGBoost 原生支持优雅，但在内存够用的抽样阶段完全可行
for col in ['Origin', 'Dest', 'Reporting_Airline']:
    df_sample[col] = df_sample[col].astype('category').cat.codes

X = df_sample[features]
y = df_sample['ArrDelay']

del df
gc.collect()

print("3. 划分数据集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 🌟 核心魔法：并行训练三个分位数模型！
# ==========================================
print("4. 开始训练不确定性区间模型...")

# 模型 1：下边界 (第 10 分位数，极度乐观预测)
print("   -> 训练下边界 (10th Quantile)...")
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100, max_depth=5, random_state=42)
model_lower.fit(X_train, y_train)

# 模型 2：核心预测 (第 50 分位数，中位数点估计)
print("   -> 训练核心预测 (50th Quantile)...")
model_median = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100, max_depth=5, random_state=42)
model_median.fit(X_train, y_train)

# 模型 3：上边界 (第 90 分位数，极度悲观预测)
print("   -> 训练上边界 (90th Quantile)...")
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100, max_depth=5, random_state=42)
model_upper.fit(X_train, y_train)

print("✅ 所有模型训练完毕！\n")

# ==========================================
# 5. 模拟终端用户的使用场景 (这正是你要在 Web App 里展示的！)
# ==========================================
# print("-" * 50)
# print("🛫 乘客查询演示 (模拟 Web App 输出):")
# print("-" * 50)

# # 我们随机从测试集中挑出 5 个航班给用户看结果
# X_demo = X_test.iloc[:5]
# y_true_demo = y_test.iloc[:5]

# # 用三个模型分别预测
# pred_lower = model_lower.predict(X_demo)
# pred_median = model_median.predict(X_demo)
# pred_upper = model_upper.predict(X_demo)

# for i in range(5):
#     print(f"航班 {i+1}:")
#     print(f"  👉 [模型点估计] 预计延误: {pred_median[i]:.0f} 分钟")
#     print(f"  👉 [不确定性区间] 我们有 80% 的把握，实际延误在 [{pred_lower[i]:.0f} 到 {pred_upper[i]:.0f}] 分钟之间。")
#     print(f"  🎯 (事后诸葛亮) 现实中这趟航班的真实延误是: {y_true_demo.iloc[i]:.0f} 分钟")
#     print()

# ==========================================
# 6. 验证集全面评估 (Uncertainty Evaluation)
# ==========================================
print("-" * 50)
print("📈 验证集 (Test Set) 统计表现:")
print("-" * 50)

# 1. 批量预测测试集
test_lower = model_lower.predict(X_test)
test_median = model_median.predict(X_test)
test_upper = model_upper.predict(X_test)

# 2. 计算覆盖率 (Coverage)
# 检查真实值 y_test 是否在 [lower, upper] 之间
in_bounds = (y_test >= test_lower) & (y_test <= test_upper)
coverage = np.mean(in_bounds) * 100

# 3. 计算平均区间宽度
avg_width = np.mean(test_upper - test_lower)

# 4. 计算中位数预测的 MAE (与之前的 XGBoost 对标)
median_mae = mean_absolute_error(y_test, test_median)

print(f"1. 区间覆盖率: {coverage:.2f}% (目标: 80.00%)")
print(f"   -> 解释: 真实航班有 {coverage:.2f}% 的概率落在模型给出的范围内。")
print(f"2. 平均区间宽度: {avg_width:.2f} 分钟")
print(f"   -> 解释: 预测的不确定性范围平均约为 {avg_width:.2f} 分钟。")
print(f"3. 点估计 (Median) MAE: {median_mae:.2f} 分钟")

# ==========================================
# 7. 评估不平衡性 (Optional but useful for Report)
# ==========================================
# 看看实际值超出上边界的比例 (应该是 10% 左右)
overshoot = np.mean(y_test > test_upper) * 100
# 看看实际值低于下边界的比例 (应该是 10% 左右)
undershoot = np.mean(y_test < test_lower) * 100

print(f"\n4. 异常情况分析:")
print(f"   - 实际延误超过预测上界的航班 (极度晚点): {overshoot:.2f}%")
print(f"   - 实际延误低于预测下界的航班 (极其提前): {undershoot:.2f}%")