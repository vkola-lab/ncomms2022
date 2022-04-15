import csv

nacc, adni, fhs = 0, 0, 0
with open('ALL.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if not row['diff_days']: continue
        if row['dataset'] == 'NACC':
            if int(row['diff_days']) < 365 * 2:
                nacc += 1
        if row['dataset'] == 'ADNI':
            if int(row['diff_days']) < 365 * 2:
                adni += 1
        if row['dataset'] == 'FHS':
            if int(row['diff_days']) < 365 * 2:
                fhs += 1

print(nacc, adni, fhs)
print(nacc + adni + fhs)