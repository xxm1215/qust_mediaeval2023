import json

path = "expanded_dataset_0.0003.json"


c = 0
a = 0
full = 0
with open(path, 'r') as inf:
    data = json.load(inf)
    for line in data:
        full += 1
        # print(line)
        if line['label'] == 1:
           c += 1
        else:
           a += 1
print(c)
print(a)
print(full)
