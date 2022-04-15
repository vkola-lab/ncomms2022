import os
from collections import defaultdict
from datetime import date
import csv

file_list = defaultdict(list)
for root, dirs, files in os.walk('/path when you unzip PPMI.zip/', topdown=False):
	for file in files:
		if file.endswith('.npy'):
			id = file.split('_')[1]
			file_list[id].append(os.path.join(root, file))

print(len(file_list))

dict_data = []

lookup = {0:'BL', 3:'V01', 6:'V02', 9:'V03', 12:'V04', 18:'V05', 24:'V06', 30:'V07', 36:'V08', 42:'V09', 48:'V10', 54:'V11',
          60:'V12', 72:'V13', 84:'V14', 96:'V15'}
lookuplist = list(lookup.keys())


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

for key in file_list:
    pool = []
    for file in file_list[key]:
        filename = file.split('/')[-1]
        da = file.split('/')[-3].split('_')[0]
        pool.append((da, filename))
    pool.sort()
    base = date.fromisoformat(pool[0][0])
    dict_data.append({'filename':pool[0][1], 'EVENT_ID':'BL'})
    for da, name in pool[1:]:
        diff = date.fromisoformat(da) - base
        dif = closest(lookuplist, diff.days / 30)
        dict_data.append({'filename':name, 'EVENT_ID':lookup[dif]})

with open('PPMI_scans_time.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'EVENT_ID'])
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)



