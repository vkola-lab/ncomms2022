import csv

LBD, DLB, PDD = 0, 0, 0
with open('/home/sq/multi-task/lookupcsv/dataset_table/NACC_ALL/NACC.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['AD'] == '0' and row['DLB'] == '1':
            DLB += 1
        if row['AD'] == '0' and row['LBD'] == '1':
            LBD += 1
        if row['AD'] == '0' and row['PDD'] == '1':
            PDD += 1
print("LBD", LBD)
print("DLB", DLB)
print("PDD", PDD)


