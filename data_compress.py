import pandas as pd
import glob
import os

folder_path = r"D:\文案\UWM\628 Data Science Practicum\module3\data"
file_list = glob.glob(os.path.join(folder_path, "*_clean.csv"))

# 🌟 加上 Cancelled 列
columns_to_keep = [
    'Month', 'DayofMonth', 'DayOfWeek', 'Distance', 
    'CRSDepTime', 'CRSArrTime', 'ArrDelay',
    'Origin', 'Dest', 'Reporting_Airline',
    'Cancelled',
    'DepDelay',   # <--- 新增：出发延误
    'TaxiOut',    # <--- 新增：起飞前滑行时间
    'TaxiIn'  # <--- 新增！
    
]

dtype_dict = {
    'Origin': 'category',
    'Dest': 'category',
    'Reporting_Airline': 'category',
    'Month': 'int32',
    'DayofMonth': 'int32',
    'DayOfWeek': 'int32',
    'Distance': 'float32',
    'CRSDepTime': 'float32', 
    'CRSArrTime': 'float32',
    'ArrDelay': 'float32',
    'Cancelled': 'float32', # <--- 新增！给它加上内存压缩
    'DepDelay': 'float32', # <--- 新增
    'TaxiOut': 'float32',  # <--- 新增
    'TaxiIn': 'float32'
}

# 下面的循环读取和保存代码完全保持不变...

df_list = []
for file in file_list:
    print(f"读取并深度优化: {os.path.basename(file)}")
    temp_df = pd.read_csv(file, usecols=columns_to_keep, dtype=dtype_dict)
    df_list.append(temp_df)

print("正在合并数据...")
df_full = pd.concat(df_list, ignore_index=True)

# 确保 concat 后这三列依然是 category 属性
for col in ['Origin', 'Dest', 'Reporting_Airline']:
    df_full[col] = df_full[col].astype('category')

# 保存为新的极小体积 Parquet
output_path = os.path.join(folder_path, "merged_airline_data_optimized.parquet")
print("正在保存终极优化的 Parquet 文件...")
df_full.to_parquet(output_path)
print(f"✅ 终极打包完成！新文件为: {output_path}")