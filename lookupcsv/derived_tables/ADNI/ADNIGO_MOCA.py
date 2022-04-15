import csv

dict_data = []
column_names = ['Phase',
                'RID',
               'VISCODE',
               'moca',
               'moca_execu',
               'moca_visuo',
               'moca_name',
               'moca_atten',
               'moca_senrep',
               'moca_verba',
               'moca_abstr',
               'moca_delrec',
               'moca_orient']

def moca_total(row):
    MOCATOT = 0
    content[column] = [row['SERIAL1'], row['SERIAL2'], row['SERIAL3'], row['SERIAL4'], row['SERIAL5']]
    if '' not in content[column]:
        SERIALSUM = sum([int(a) for a in content[column]])
        if SERIALSUM == 5 or SERIALSUM == 4:
            MOCATOT += 3
        elif SERIALSUM == 2 or SERIALSUM == 3:
            MOCATOT += 2
        elif SERIALSUM == 1:
            MOCATOT += 1
    content[column] = [row['DELW1'], row['DELW2'], row['DELW3'], row['DELW4'], row['DELW5']]
    if '' not in content[column]:
        for a in content[column]:
            if a == '1':
                MOCATOT += 1
    if row['FFLUENCY'] and int(row['FFLUENCY']) >= 11:
        MOCATOT += 1
    if row['LETTERS'] and int(row['LETTERS']) < 2:
        MOCATOT += 1
    keys = ["TRAILS", "CUBE", "CLOCKCON", "CLOCKNO", "CLOCKHAN", "LION",  "RHINO", "CAMEL", "DIGFOR",
            "DIGBACK", "REPEAT1", "REPEAT2", "ABSTRAN", "ABSMEAS", "DATE", "MONTH", "YEAR", "DAY", "PLACE", "CITY"]
    for key in keys:
        if row[key]:
            MOCATOT += int(row[key])
    return MOCATOT


with open('../../raw_tables/ADNI/MOCA.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        content = {}
        for column in column_names:
            if column in ['Phase','RID', 'VISCODE']:
                content[column] = row[column]
            elif column == 'moca_execu':
                content[column] = row['TRAILS']
            elif column == 'moca':
                content[column] = moca_total(row)
            elif column == 'moca_visuo':
                content[column] = [row['CUBE'], row['CLOCKCON'], row['CLOCKNO'], row['CLOCKHAN']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
                else:
                    content[column] = ''
            elif column == 'moca_name':
                content[column] = [row['LION'], row['RHINO'], row['CAMEL']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
                else:
                    content[column] = ''
            elif column == 'moca_atten':
                content[column] = [row['DIGFOR'], row['DIGBACK'], row['LETTERS'], row['SERIAL1'], row['SERIAL2'], row['SERIAL3'], row['SERIAL4'], row['SERIAL5']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
                else:
                    content[column] = ''
            elif column == 'moca_senrep':
                content[column] = [row['REPEAT1'], row['REPEAT2']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
            elif column == 'moca_verba':
                content[column] = row['FFLUENCY']
            elif column == 'moca_abstr':
                content[column] = [row['ABSTRAN'], row['ABSMEAS']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
                else:
                    content[column] = ''
            elif column == 'moca_delrec':
                deltec = [row['DELW1'], row['DELW2'], row['DELW3'], row['DELW4'], row['DELW5']]
                content[column] = 0
                if '' not in deltec:
                    for a in deltec:
                        if a == '1':
                            content[column] += 1
                else:
                    content[column] = ''
            elif column == 'moca_orient':
                content[column] = [row['DATE'], row['MONTH'], row['YEAR'], row['DAY'], row['PLACE'], row['CITY']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
                else:
                    content[column] = ''
        dict_data.append(content)


with open('ADNIGO_MOCA.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)