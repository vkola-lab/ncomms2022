import csv
from datetime import date
from collections import defaultdict

thres = 365 * 2

def get_all_filenames():
    content = []
    with open('NACC_NP_eval.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            content.append({'filename' : row["filename"], "dataset" : "NACC"})
    add_NACCID(content)
    with open('ADNI_NP_eval.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            content.append({'filename' : row["filename"],
                            "dataset" : "ADNI",
                            "id" : row["filename"].split("_")[3],
                            "RID" : row["filename"].split("_")[3]})
    with open('FHS_NP_eval.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            content.append({'filename' : row["filename"],
                            "dataset" : "FHS",
                            "id" : row['filename'].split('_')[0],
                            "FHSID" : row['filename'].split('_')[0],
                            "MRI_date" : parse_FHS_date(row['filename'])})
    return content

def parse_FHS_date(string):
    date = string.split('_')[-1].strip('.npy')
    year, month, day = date[:4], date[4:6], date[6:]
    return '/'.join([month, day, year])

def add_NACCID(content):
    zip_id = {}
    with open("nacc_np.csv", 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['NACCMRFI'][:4] != 'NACC':
                zip_id[row['NACCMRFI'].strip('.zip')] = row['NACCID']
    for case in content:
        if case['dataset'] == 'NACC':
            if case['filename'][:4] == 'NACC':
                case['id'] = case['filename'][:10]
            else:
                case['id'] = zip_id[case['filename'].split("_")[0]]
            case['NACCID'] = case['id']

def add_MRI_date(content):
    ADNI_rid_mridate = {}
    with open('ADNI_NP_MRI_meta.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rid = row['Subject'].split('_')[-1]
            ADNI_rid_mridate[rid] = row['Acq Date']
    NACC_filename_mridate = {}
    with open('kolachalama12042020mri.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            NACC_filename_mridate[row['NACCMRFI'].strip('.zip')] = '/'.join([row['MRIMO'], row['MRIDY'], row['MRIYR']])
    for case in content:
        if case['dataset'] == 'ADNI':
            case['MRI_date'] = ADNI_rid_mridate[case['RID']]
        if case['dataset'] == 'NACC':
            if case['filename'][:3] == 'mri':
                key = case['filename'].split('_')[0]
            elif case['filename'][:4] == 'NACC':
                # print('----------------------- ')
                # print('filename is ', case['filename'])
                key = "_".join(case['filename'].split('_')[:2])
                # print('key is ', key)
            case['MRI_date'] = NACC_filename_mridate[key]

def add_dod(content):
    FHS_id_dod = {}
    with open('FHS_death.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = row["datedth"].split('/')
            id = row["idtype"] + '-' + row["id"]
            FHS_id_dod[id] = '/'.join([month, day, year])
    ADNI_id_dod = {}
    with open('ADNI_np.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            id = row["RID"]
            ADNI_id_dod[id] = row["NPDOD"]
    NACC_id_dod = {}
    with open('kolachalama12042020.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            NACC_id_dod[row['NACCID']] = "/".join([row["NACCMOD"], '1', row['NACCYOD']])
    for case in content:
        if case['dataset'] == 'FHS':
            key = case['id'].split('-')
            key = "-".join([key[0], str(int(key[1]))])
            if key in FHS_id_dod:
                case['Death_date'] = FHS_id_dod[key]
        if case['dataset'] == 'ADNI':
            key = str(int(case['RID']))
            if key in ADNI_id_dod:
                case['Death_date'] = ADNI_id_dod[key]
        if case['dataset'] == 'NACC':
            key = case['NACCID']
            if key in NACC_id_dod:
                case['Death_date'] = NACC_id_dod[key]

def add_diffdays(content):
    for case in content:
        if 'MRI_date' in case and case['MRI_date'] and 'Death_date' in case and case['Death_date']:
            mri_date = case['MRI_date'].split('/')
            np_date = case['Death_date'].split('/')
            mri_date, np_date = list(map(int, mri_date)), list(map(int, np_date))
            date1 = date(mri_date[2], mri_date[0], mri_date[1])
            date2 = date(np_date[2], np_date[0], np_date[1])
            case['diff_days'] = (date2 - date1).days

def add_abc(content):
    NACC_id_abc = {}
    with open('nacc_np.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            NACC_id_abc[row['NACCID']] = [A_(row['NPTHAL']), B_(row['NACCBRAA']), C_(row['NACCNEUR'])]
    ADNI_id_abc = {}
    with open('ADNI_np.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            ADNI_id_abc[row['RID']] = [A_(row['NPTHAL']), B_(row['NPBRAAK']), C_(row['NPNEUR'])]
    FHS_id_abc = {}
    with open('FHS_ADNI-NPATH_05-31-21.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['framid']:
                key = row['idtype'] + '-' + row['id']
                FHS_id_abc[key] = [A_(row['NPDIFF']), B_(row['NPBRAAK']), C_(row['NPNEUR'])]
    for case in content:
        if case['dataset'] == 'NACC' and case['NACCID'] in NACC_id_abc:
            case['A_score'], case['B_score'], case['C_score'] = NACC_id_abc[case['NACCID']]
        if case['dataset'] == 'ADNI':
            key = str(int(case['RID']))
            if key in ADNI_id_abc:
                case['A_score'], case['B_score'], case['C_score'] = ADNI_id_abc[key]
        if case['dataset'] == 'FHS':
            key = case['FHSID']
            key = key.split('-')
            key = key[0] + '-' + str(int(key[1]))
            if key in FHS_id_abc:
                case['A_score'], case['B_score'], case['C_score'] = FHS_id_abc[key]
            # else:
            #     print('key not found', key)

def A_(raw):
    if raw == '0': return 0
    if raw in ['1', '2']: return 1
    if raw in ['3']: return 2
    if raw in ['4', '5']: return 3
    return ''

def B_(raw):
    if raw == '0': return 0
    if raw in ['1', '2']: return 1
    if raw in ['3', '4']: return 2
    if raw in ['5', '6']: return 3
    return ''

def C_(raw):
    if raw == '0': return 0
    if raw == '1': return 1
    if raw == '2': return 2
    if raw in ['3', '4']: return 3
    return ''

def add_CNN_scores(content):
    scores = {}
    with open('NACC_NP_eval.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scores[row['filename']] = [row['COG_score'], row['ADD_score']]
    with open('ADNI_NP_eval.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scores[row['filename']] = [row['COG_score'], row['ADD_score']]
    with open('FHS_NP_eval.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scores[row['filename']] = [row['COG_score'], row['ADD_score']]
    for case in content:
        if case['filename'] in scores:
            case['CNN_COG_score'], case['CNN_ADD_prob'] = scores[case['filename']]

def add_region_neuropath(content):
    prefixes = ['CG_1', 'FL_mfg_7', 'FL_pg_10', 'TL_stg_32', 'PL_ag_20', 'Amygdala_24',
                'TL_hippocampus_28', 'TL_parahippocampal_30', 'c_37', 'bs_35', 'sn_48', 'th_49',
                'pal_46', 'na_45X', 'cn_36C', 'OL_17_18_19OL']
    whole = ['AB_DP', 'TAU_NFT', 'TAU_NP', 'SILVER_NFT']
    columns = [p + '_' + s for p in prefixes for s in whole]
    ADNI_rid_np = {}
    with open('ADNI_np.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            ADNI_rid_np[row['RID']] = [row[key] if row[key]!='9' else '' for key in columns]
    FHS_id_np = {}
    with open('FHS_ADNI-NPATH_05-31-21.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['framid']:
                key = row['idtype'] + '-' + row['id']
                FHS_id_np[key] = [row[k] for k in columns]
    for case in content:
        if case['dataset'] == 'ADNI':
            rid = str(int(case['RID']))
            if rid in ADNI_rid_np:
                assert len(columns) == len(ADNI_rid_np[rid]), 'ADNI length mismatch'
                for col, val in zip(columns, ADNI_rid_np[rid]):
                    case[col] = val
        if case['dataset'] == 'FHS':
            key = case['FHSID']
            key = key.split('-')
            id = key[0] + '-' + str(int(key[1]))
            if id in FHS_id_np:
                assert len(columns) == len(FHS_id_np[id]), 'FHS length mismatch'
                for col, val in zip(columns, FHS_id_np[id]):
                    case[col] = val

def add_diagnosis(content):
    ADNI_id_diagnosis = {}
    with open('ADNI_NP_MRI_meta.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rid = row['Subject'].split('_')[-1]
            ADNI_id_diagnosis[rid] = row['Group']
    NACC_id_diagnosis = defaultdict(dict)
    with open('NACCdiagTable.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            date = "/".join([row['VISITMO'], row['VISITDAY'], row['VISITYR']])
            label = ''
            if row['COG'] == '0':
                label = 'NC'
            elif row['COG'] == '1':
                label = 'MCI'
            elif row['COG'] == '2':
                if row['AD'] == '1':
                    label = 'ADD'
                elif row['AD'] == '0':
                    label = 'nADD'
            NACC_id_diagnosis[row['NACCID']][date] = label
    FHS_id_diagnosis = defaultdict(list)
    with open('FHS_dementia_reviews.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            id = row['idtype'] + '-' + '0'* (4-len(row['id'])) + row['id']
            l = [row['normal_date'], row['impairment_date'], row['mild_date'], row['moderate_date'], row['severe_date'], row['demrv103']]
            FHS_id_diagnosis[id] = l
    for case in content:
        if case['dataset'] == 'ADNI':
            if case['RID'] in ADNI_id_diagnosis:
                if ADNI_id_diagnosis[case['RID']] == 'CN':
                    case['Group'] = 'NC'
                elif 'MCI' in ADNI_id_diagnosis[case['RID']]:
                    case['Group'] = 'MCI'
                elif ADNI_id_diagnosis[case['RID']] == 'AD':
                    case['Group'] = 'ADD'
                else:
                    case['Group'] = ADNI_id_diagnosis[case['RID']]
            else:
                print('ADNI key not found', case['RID'])
        elif case['dataset'] == 'NACC':
            if case['NACCID'] in NACC_id_diagnosis:
                label = find_closest(case['MRI_date'], NACC_id_diagnosis[case['NACCID']])
                case['Group'] = label
        elif case['dataset'] == 'FHS':
            if case['FHSID'] in FHS_id_diagnosis:
                case['Group'] = get_FHS_label(case['MRI_date'], FHS_id_diagnosis[case['FHSID']])

def get_FHS_label(mri_date, info):
    days_to = []
    for i in range(5):
        if info[i]:
            year, month, day = info[i].split('-')
            info[i] = '/'.join([month, day, year])
            days_to.append(get_days(mri_date, info[i]))
        else:
            days_to.append('')
    interval = days_to[:]
    for i in range(5):
        if not interval[i]:
            interval[i] = 100000
        else:
            if i == 0:
                interval[0] = 0 if int(interval[0]) > 0 else abs(interval[0])
            else:
                interval[i] = abs(interval[i])
    index = interval.index(min(interval))
    value = interval[index]
    if value > thres:
        return ''
    if index == 0:
        return 'NC'
    elif index == 1:
        return 'MCI'
    else:
        if info[-1] in ['1', '2', '4']:
            return 'ADD'
        else:
            return 'nADD'


def get_days(mri, v):
    mri_date = mri.split('/')
    visit = v.split('/')
    mri_date, visit = list(map(int, mri_date)), list(map(int, visit))
    date1 = date(mri_date[2], mri_date[0], mri_date[1])
    date2 = date(visit[2], visit[0], visit[1])
    return (date2 - date1).days

def find_closest(mri, book):
    pool = []
    for v in book:
        if book[v]:
            mri_date = mri.split('/')
            visit = v.split('/')
            mri_date, visit = list(map(int, mri_date)), list(map(int, visit))
            date1 = date(mri_date[2], mri_date[0], mri_date[1])
            date2 = date(visit[2], visit[0], visit[1])
            diff_days = abs((date2 - date1).days)
            pool.append((diff_days, book[v]))
    pool.sort()
    if pool and pool[0][0] < thres:
        return pool[0][1]
    else:
        return ''


def write_content(content):
    prefixes = ['CG_1', 'FL_mfg_7', 'FL_pg_10', 'TL_stg_32', 'PL_ag_20', 'Amygdala_24',
                'TL_hippocampus_28', 'TL_parahippocampal_30', 'c_37', 'bs_35', 'sn_48', 'th_49',
                'pal_46', 'na_45X', 'cn_36C', 'OL_17_18_19OL']
    whole = ['AB_DP', 'TAU_NFT', 'TAU_NP', 'SILVER_NFT']
    columns = [p + '_' + s for p in prefixes for s in whole]
    fieldnames = ['filename', 'dataset', 'id', 'RID', 'FHSID', 'NACCID', 'MRI_date', 'Death_date', 'diff_days',
                  'Group', 'A_score', 'B_score', 'C_score', 'CNN_COG_score', 'CNN_ADD_prob'] + columns
    with open('ALL.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for case in content:
            writer.writerow(case)




if __name__ == "__main__":
    content = get_all_filenames()
    add_MRI_date(content)
    add_dod(content)
    add_diffdays(content)
    add_diagnosis(content)
    add_abc(content)
    add_CNN_scores(content)
    add_region_neuropath(content)
    write_content(content)


