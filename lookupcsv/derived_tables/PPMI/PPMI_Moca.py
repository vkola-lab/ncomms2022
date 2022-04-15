import csv

dict_data = []
column_names = ['REC_ID',
               'F_STATUS',
               'PATNO',
               'EVENT_ID',
               'PAG_NAME',
               'INFODT',
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

with open('../../raw_tables/PPMI/Montreal_Cognitive_Assessment__MoCA_.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        content = {}
        for column in column_names:
            if column in ['REC_ID','F_STATUS','PATNO','EVENT_ID','PAG_NAME','INFODT']:
                content[column] = row[column]
            elif column == 'moca_execu':
                content[column] = row['MCAALTTM']
            elif column == 'moca':
                content[column] = row['MCATOT']
            elif column == 'moca_visuo':
                content[column] = [row['MCACUBE'], row['MCACLCKC'], row['MCACLCKN'], row['MCACLCKH']]
                content[column] = sum([int(a) for a in content[column]])
            elif column == 'moca_name':
                content[column] = [row['MCALION'], row['MCARHINO'], row['MCACAMEL']]
                content[column] = sum([int(a) for a in content[column]])
            elif column == 'moca_atten':
                content[column] = [row['MCAFDS'], row['MCABDS'], row['MCAVIGIL'], row['MCASER7']]
                content[column] = sum([int(a) for a in content[column]])
            elif column == 'moca_senrep':
                content[column] = row['MCASNTNC']
            elif column == 'moca_verba':
                content[column] = row['MCAVF']
            elif column == 'moca_abstr':
                content[column] = row['MCAABSTR']
            elif column == 'moca_delrec':
                content[column] = [row['MCAREC1'], row['MCAREC2'], row['MCAREC3'], row['MCAREC4'], row['MCAREC5']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
            elif column == 'moca_orient':
                content[column] = [row['MCADATE'], row['MCAMONTH'], row['MCAYR'], row['MCADAY'], row['MCAPLACE'], row['MCACITY']]
                if '' not in content[column]:
                    content[column] = sum([int(a) for a in content[column]])
        dict_data.append(content)

with open('PPMI_MoCa.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)