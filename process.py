import json

# Reading raw json file
with open('Tabular_INPUT_new.json', 'r') as f:
    data = json.load(f)

# Find the minimum and maximum values ​​of YearBuilt and EstimatedValue
min_year_built = min(item['YearBuilt'] for item in data)

max_estimated_value = max(item['EstimatedValue'] for item in data)
for item in data:
    item['YearBuilt'] -= min_year_built
    # Classify the EstimatedValue according to multiples of 200000
    item['EstimatedValue_level'] = int(item['EstimatedValue'] / 200000)
    item['dist_track_line'] = int(item['dist_track_line'] / 500)
    item['dist_track_landfall'] = int(item['dist_track_landfall'] / 500)
    item['wind_mean'] = int(item['wind_mean'] / 1)
    item['flood_mean'] = int(item['flood_mean'] / 1)

with open('data.json', 'w') as f:
    json.dump(data, f, indent=4)

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
