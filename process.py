import json

# 读取原始json文件
with open('Tabular_INPUT_new.json', 'r') as f:
    data = json.load(f)

# 找到YearBuilt和EstimatedValue的最小值和最大值
min_year_built = min(item['YearBuilt'] for item in data)

max_estimated_value = max(item['EstimatedValue'] for item in data)

# 对每个字典进行处理
for item in data:
    # 减去最小的YearBuilt值
    item['YearBuilt'] -= min_year_built
    # 根据200000的倍数对EstimatedValue进行分级
    item['EstimatedValue_level'] = int(item['EstimatedValue'] / 200000)
    item['dist_track_line'] = int(item['dist_track_line'] / 500)
    item['dist_track_landfall'] = int(item['dist_track_landfall'] / 500)
    item['wind_mean'] = int(item['wind_mean'] / 1)
    item['flood_mean'] = int(item['flood_mean'] / 1)
# 保存新的json文件
with open('data.json', 'w') as f:
    json.dump(data, f, indent=4)

# 保存YearBuilt和EstimatedValue的最大值
result = {
    'max_year_built': max(item['YearBuilt']+1 for item in data),
    'max_estimated_value': max(item['EstimatedValue_level']+1 for item in data),
    'max_Evacuation_Zone': max(item['Evacuation_Zone']+1 for item in data),
    'max_dist_track_line': max(item['dist_track_line']+1 for item in data),
    'max_dist_track_landfall': max(item['dist_track_landfall']+1 for item in data),
    'max_wind_mean': max(item['wind_mean']+1 for item in data),
    'max_flood_mean': max(item['flood_mean']+1 for item in data),
}

with open('max_values.json', 'w') as f:
    json.dump(result, f, indent=4)
