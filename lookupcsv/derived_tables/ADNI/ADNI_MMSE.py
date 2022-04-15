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
               'Orientation',
               'Working Memory',
               'Concentration',
               'Memory Recall',
               'Language',
               'Visuospatial',
               'mmse']

with open('../raw_tables/ADNI_MMSE.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        content = {}
        for column in column_names:
            if column in ['Phase', 'ID', 'RID', 'SITEID', 'VISCODE', 'VISCODE2', 'USERDATE', 'USERDATE2', 'EXAMDATE']:
                content[column] = row[column]
            elif column == 'Orientation':
                content[column] = [row['MMDATE'], row['MMYEAR'], row['MMMONTH'], row['MMDAY'], row['MMSEASON'],
                                   row['MMHOSPIT'], row['MMFLOOR'], row['MMCITY'], row['MMAREA'], row['MMSTATE']]
                print(content[column])
                content[column] = sum([int(a) - 1 for a in content[column] if a in ['1', '2'] ]) # correct is 0, incorrect is 1 for each item, the higher the worse
            elif column == 'Working Memory':
                content[column] = [row['MMBALL'], row['MMFLAG'], row['MMTREE'], row['MMTRIALS']]
                content[column] = sum([int(a) - 1 for a in content[column] if a not in ['-1', ''] ])  # correct is 0, incorrect is 1 for each item, the higher the worse
            elif column == 'Concentration':
                content[column] = [row['MMD'], row['MML'], row['MMR'], row['MMO'], row['MMW']]
                content[column] = sum([int(a) - 1 for a in content[column] if a in ['1', '2'] ])  # correct is 0, incorrect is 1 for each item, the higher the worse
            elif column == 'Memory Recall':
                content[column] = [row['MMBALLDL'], row['MMFLAGDL'], row['MMTREEDL']]
                content[column] = sum([int(a) - 1 for a in content[column] if a in ['1', '2'] ])  # correct is 0, incorrect is 1 for each item, the higher the worse
            elif column == 'Language':
                content[column] = [row['MMWATCH'], row['MMPENCIL'], row['MMREPEAT'], row['MMHAND'],
                                   row['MMFOLD'], row['MMONFLR'], row['MMREAD'], row['MMWRITE']]
                content[column] = sum([int(a) - 1 for a in content[column] if a in ['1', '2'] ])  # correct is 0, incorrect is 1 for each item, the higher the worse
            elif column == 'Visuospatial':
                content[column] = [row['MMDRAW']]
                content[column] = sum([int(a) - 1 for a in content[column] if a in ['1', '2'] ])  # correct is 0, incorrect is 1 for each item, the higher the worse
            elif column == 'mmse':
                content[column] = [row['MMSCORE']]
                content[column] = sum([int(a) for a in content[column] if a not in ['-1', ''] ])  # correct is 0, incorrect is 1 for each item, the higher the worse
        dict_data.append(content)

with open('ADNI_MMSE.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)















