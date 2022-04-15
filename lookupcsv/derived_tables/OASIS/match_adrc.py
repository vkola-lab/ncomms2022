import csv
from collections import defaultdict
import datetime
import os
from glob import glob

ADRCTable = '../../raw_tables/OASIS/ADRC_Clinical_Data.csv'

ADRCMap = defaultdict(dict)
with open(ADRCTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['ADRC_ADRCCLINICALDATA ID'].split('_')[0], row['ADRC_ADRCCLINICALDATA ID'].split('_')[2]
        result = (row['mmse'], row['cdr'], row['sumbox'])
        ADRCMap[id][date] = result


GDSTable = '../../raw_tables/OASIS/OASIS_GDS.csv'

GDSMap = defaultdict(dict)
with open(GDSTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['UDS_B6BEVGDSDATA ID'].split('_')[0], row['UDS_B6BEVGDSDATA ID'].split('_')[2]
        result = row['GDS']
        GDSMap[id][date] = result


psychTable = '../../raw_tables/OASIS/OASIS_psychAssessment.csv'

psychMap = defaultdict(dict)
with open(psychTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['CNDA_PSYCHOMETRICSDATA ID'].split('_')[0], row['CNDA_PSYCHOMETRICSDATA ID'].split('_')[2]
        result = (row['LOGIMEM'], row['MEMUNITS'], row['DIGIF'], row['DIGIFLEN'], row['DIGIB'], row['DIGIBLEN'], row['ANIMALS'], row['BOSTON'], row['TRAILA'], row['TRAILB'])
        psychMap[id][date] = result


NPITable = '../../raw_tables/OASIS/OASIS_NPI-Q.csv'

vari_list = ['DEL', 'HALL', 'AGIT', 'DEPD', 'ANX', 'ELAT', 'APA', 'DISN', 'IRR', 'MOT', 'NITE', 'APP']
NPIMap = defaultdict(dict)
with open(NPITable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['UDS_B5BEHAVASDATA ID'].split('_')[0], row['UDS_B5BEHAVASDATA ID'].split('_')[2]
        result = [row[vari] for vari in vari_list] + [row[vari + 'SEV'] for vari in vari_list]
        NPIMap[id][date] = result


FAQTable = '../../raw_tables/OASIS/OASIS_UDS_B7_FAQs.csv'
vari_list_ = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
FAQMap = defaultdict(dict)
with open(FAQTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['UDS_B7FAQDATA ID'].split('_')[0], row['UDS_B7FAQDATA ID'].split('_')[2]
        result = [row[vari] for vari in vari_list_]
        FAQMap[id][date] = result


HISTable = '../../raw_tables/OASIS/OASIS_health_history.csv'
vari_list_his = ['CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA',
              'SEIZURES', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF',
              'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR', 'PSYCDIS', 'ALCOHOL', 'TOBAC100',
              'SMOKYRS', 'PACKSPER', 'ABUSOTHR']
HISMap = defaultdict(dict)
with open(HISTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['UDS_A5SUBHSTDATA ID'].split('_')[0], row['UDS_A5SUBHSTDATA ID'].split('_')[2]
        result = [row[vari] for vari in vari_list_his]
        HISMap[id][date] = result


MRI_list = glob("/data_2/OASIS/*_*/")
mriMap = defaultdict(dict)

for mri in MRI_list:
    id, date = mri.split('/')[-2].split('_')[0], mri.split('/')[-2].split('_')[2]
    mriMap[id][date] = mri

def find_closest(id, date, Map):
    min_diff = 1000000
    ans = None
    mriDate = int(date.strip('d'))
    diagTimes = Map[id].keys()
    for time in diagTimes:
        diagDate = int(time.strip('d'))
        diff = diagDate - mriDate
        diff = abs(diff)
        if diff < min_diff and diff < 183:  # within +/- 6 months
            min_diff = diff
            ans = time
    return ans

content = []

for mri in MRI_list:
    row = [mri]
    id, date = mri.split('/')[-2].split('_')[0], mri.split('/')[-2].split('_')[2]
    ADRCDate = find_closest(id, date, ADRCMap)
    GDSDate = find_closest(id, date, GDSMap)
    psychDate = find_closest(id, date, psychMap)
    NPIDate = find_closest(id, date, NPIMap)
    FAQDate = find_closest(id, date, FAQMap)
    HISDate = find_closest(id, date, HISMap)
    row.append(id)
    if ADRCDate:
        row.extend(ADRCMap[id][ADRCDate])
    else:
        row.extend(['']*3)
    if GDSDate:
        row.append(GDSMap[id][GDSDate])
    else:
        row.append('')
    if psychDate:
        row.extend(psychMap[id][psychDate])
    else:
        row.extend(['']*10)
    if NPIDate:
        row.extend(NPIMap[id][NPIDate])
    else:
        row.extend(['']*24)
    if FAQDate:
        row.extend(FAQMap[id][FAQDate])
    else:
        row.extend(['']*10)
    if HISDate:
        row.extend(HISMap[id][HISDate])
    else:
        row.extend(['']*28)
    content.append(row)

content = sorted(content, key=lambda x:x[0])

features = ['mmse', 'cdr', 'cdrSum', 'gds', 'lm_imm', 'lm_del', 'digitF', 'digitFL', 'digitB', 'digitBL', 'animal', 'boston', 'trailA', 'trailB'] + \
           vari_list + [a+'SEV' for a in vari_list] + vari_list_ + vari_list_his

with open('mri_adrc_table_6months.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['folderName', 'ID']+features)
    writer.writeheader()
    case = {}
    for row in content:
        case['folderName'] = row[0]
        case['ID'] = row[1]
        for i, vari in enumerate(features):
            case[vari] = row[2+i]
        print(case)
        writer.writerow(case)
