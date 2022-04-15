import csv

dict_data = []
column_names = ['Phase',
               'ID',
               'RID',
               'SITEID',
               'VISCODE',
               'VISCODE2',
               'USERDATE',
               'USERDATE2',
               'EXAMDATE',
               'Diagnosis']


with open('../raw_tables/ADNI_DXSUM_PDXCONV.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        content = {}
        for column in column_names:
            if column in ['Phase', 'ID', 'RID', 'SITEID', 'VISCODE', 'VISCODE2', 'USERDATE', 'USERDATE2', 'EXAMDATE']:
                content[column] = row[column]
            elif column == 'Diagnosis':
                if row['DXCURREN'] == '1': # normal
                    content[column] = 'NL'
                    dict_data.append(content)
                elif row['DXCURREN'] == '2': # use all type of mci
                    content[column] = 'MCI'
                    dict_data.append(content)
                elif row['DXCURREN'] == '3': # dementia
                    if row['DXAD'] == '1': # etiology is AD
                        content[column] = 'AD'
                        dict_data.append(content)

with open('ADNI_Diagnosis.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)

