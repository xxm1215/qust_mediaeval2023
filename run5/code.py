import json
from collections import defaultdict

# Load JSON files
json_files = ['/home/qd/PycharmProjects/MediaEval2022/all_team2/submission/RT1_result.json',
              '/home/qd/PycharmProjects/MediaEval2022/all_team2/submission/RT2_result.json', 
              '/home/qd/PycharmProjects/MediaEval2022/all_team2/submission/RT4_result.json']

# Combine data from all JSON files
combined_data = defaultdict(list)

for file in json_files:
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            combined_data[key].extend(value)

# Aggregate similarity values for the same paths
aggregated_data = {}
for key, value in combined_data.items():
    path_similarity_dict = defaultdict(float)
    for entry in value:
        path = entry['path']
        similarity = entry['similarity']
        path_similarity_dict[path] += similarity
    aggregated_data[key] = [{'path': path, 'similarity': sim} for path, sim in path_similarity_dict.items()]

# Sort paths based on aggregated similarity values and take top 100
rankings = {}
for key, value in aggregated_data.items():
    sorted_paths = sorted(value, key=lambda x: x['similarity'], reverse=True)[:100]
    rankings[key] = sorted_paths

# Save the top 100 rankings to a new JSON file
output_file = '/home/qd/PycharmProjects/MediaEval2022/all_team2/submission/RT_rangking.json'
with open(output_file, 'w') as json_output:
    json.dump(rankings, json_output)

print(f"Top 100 rankings saved to {output_file}")
