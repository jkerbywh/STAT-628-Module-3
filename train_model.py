import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 🌟 注意这里换成了刚才生成的新文件名
data_path = r"D:\文案\UWM\628 Data Science Practicum\module3\data\merged_airline_data_optimized.parquet"

print("1. 正在丝滑读取已优化的数据...")
df = pd.read_parquet(data_path)
print(f"读取成功！总行数: {len(df)}")

print("2. 抽样 100% 并清理空值...")
# df = df.sample(frac=0.1, random_state=42)
df = df.dropna(subset=['ArrDelay'])

features = [
    'Month', 'DayofMonth', 'DayOfWeek', 'Distance', 'CRSDepTime', 'CRSArrTime',
    'Origin', 'Dest', 'Reporting_Airline'
]
X = df[features]
y = df['ArrDelay']

print("3. 划分数据集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. 训练 XGBoost 模型 (支持原生 Category 特征)...")
model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=7,
    learning_rate=0.1,
    tree_method='hist',
    enable_categorical=True,  # 开启原生类别特征支持
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ 模型训练完毕！")

# 评估
y_pred = model.predict(X_test)
print("-" * 30)
print(f"MAE  (平均绝对误差): {mean_absolute_error(y_test, y_pred):.2f} 分钟")
print(f"RMSE (均方根误差): {mean_squared_error(y_test, y_pred, squared=False):.2f} 分钟")