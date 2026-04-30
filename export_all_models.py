import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import os
import gc

# ==========================================
# 1. 路径配置与全局设置
# ==========================================
data_path = r"D:\文案\UWM\628 Data Science Practicum\module3\data\merged_airline_data_optimized.parquet"
model_dir = r"D:\文案\UWM\628 Data Science Practicum\module3\models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print("🚀 [Step 1] 正在读取全量数据...")
df = pd.read_parquet(data_path)

# ==========================================
# 2. 类别编码与映射保存 (Web App 必需)
# ==========================================
print("🔄 [Step 2] 正在进行类别特征编码...")
cat_cols = ['Origin', 'Dest', 'Reporting_Airline']
category_mappings = {}

for col in cat_cols:
    df[col] = df[col].astype('category')
    category_mappings[col] = {cat: code for code, cat in enumerate(df[col].cat.categories)}
    df[col] = df[col].cat.codes

joblib.dump(category_mappings, os.path.join(model_dir, 'category_mappings.joblib'))
print("   ✅ 类别映射字典已保存！")

# 核心特征输入 (排除事后才知道的变量如 TaxiIn/Out，以防数据泄露)
features = ['Month', 'DayofMonth', 'DayOfWeek', 'Distance', 'CRSDepTime', 'CRSArrTime', 'Origin', 'Dest', 'Reporting_Airline']

# ==========================================
# 3. 训练【航班取消预测】模型 (纯 XGBoost)
# ==========================================
print("\n🚨 [Step 3] 正在训练：取消率预测模型 (XGBClassifier)...")
df_cancel = df.dropna(subset=['Cancelled'])
model_cancel = xgb.XGBClassifier(n_estimators=150, max_depth=6, tree_method='hist', random_state=42)
model_cancel.fit(df_cancel[features], df_cancel['Cancelled'])
joblib.dump(model_cancel, os.path.join(model_dir, 'model_cancel.joblib'))
print("   ✅ 取消预测模型已保存。")

del df_cancel
gc.collect()

# ==========================================
# 4. 训练【到达延误】不确定性模型 (HistGBDT 加速版)
# ==========================================
print("\n🛬 [Step 4] 正在训练：到达延误不确定性模型 (ArrDelay)...")
# 采用 5% 抽样（约30万行），在保证统计分布精确度的同时，将速度提升数十倍
df_delay_arr = df.dropna(subset=['ArrDelay']).sample(frac=0.05, random_state=42)
X_arr = df_delay_arr[features]
y_arr = df_delay_arr['ArrDelay']

quantiles = {'low': 0.1, 'med': 0.5, 'upp': 0.9}
for name, q in quantiles.items():
    print(f"   -> 正在训练 ArrDelay {name} (alpha={q})...")
    m = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=100, max_depth=5, random_state=42)
    m.fit(X_arr, y_arr)
    joblib.dump(m, os.path.join(model_dir, f'model_arr_delay_{name}.joblib'))

# 附加：训练纯 XGBoost 到达基线模型
model_xgb_arr = xgb.XGBRegressor(n_estimators=100, max_depth=6, tree_method='hist', random_state=42)
model_xgb_arr.fit(X_arr, y_arr)
joblib.dump(model_xgb_arr, os.path.join(model_dir, 'model_arr_delay_xgb_baseline.joblib'))

del df_delay_arr, X_arr, y_arr
gc.collect()

# ==========================================
# 5. 训练【出发延误】不确定性模型 (HistGBDT 加速版)
# ==========================================
print("\n🛫 [Step 5] 正在训练：出发延误不确定性模型 (DepDelay)...")
df_delay_dep = df.dropna(subset=['DepDelay']).sample(frac=0.05, random_state=42)
X_dep = df_delay_dep[features]
y_dep = df_delay_dep['DepDelay']

for name, q in quantiles.items():
    print(f"   -> 正在训练 DepDelay {name} (alpha={q})...")
    m = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=100, max_depth=5, random_state=42)
    m.fit(X_dep, y_dep)
    joblib.dump(m, os.path.join(model_dir, f'model_dep_delay_{name}.joblib'))

# 附加：训练纯 XGBoost 出发基线模型
model_xgb_dep = xgb.XGBRegressor(n_estimators=100, max_depth=6, tree_method='hist', random_state=42)
model_xgb_dep.fit(X_dep, y_dep)
joblib.dump(model_xgb_dep, os.path.join(model_dir, 'model_dep_delay_xgb_baseline.joblib'))

del df_delay_dep, X_dep, y_dep
gc.collect()

# ==========================================
# 6. 完工总结
# ==========================================
print("\n" + "="*50)
print("🎊 恭喜！所有工业级模型已训练并入库！")
print(f"📂 保存位置: {model_dir}")
print("包含的核心文件：")
print("  - category_mappings.joblib (前端解析字典)")
print("  - model_cancel.joblib (取消预警)")
print("  - model_arr_delay_low/med/upp.joblib (到达不确定性区间)")
print("  - model_dep_delay_low/med/upp.joblib (出发不确定性区间)")
print("="*50)