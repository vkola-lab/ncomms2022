import csv
from glob import glob
from collections import defaultdict


class TableData:
    def __init__(self):
        self.datasetName = 'OASIS'
        self.imageDir = '/data_2/OASIS_/npy/'
        self.imageFileNameList, self.RIDList = self.get_filenames_and_IDs(self.imageDir)
        self.columnNames = []
        self.content = defaultdict(dict) # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.npy')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        IDList = [filename.split('_')[0].strip('sub-') for filename in fileNameList]
        return fileNameList, IDList

    def addColumns_path_filename(self):
        self.columnNames.extend(['path', 'filename'])
        for idx, id in enumerate(self.RIDList):
            self.content[id]['path'] = self.imageDir
            self.content[id]['filename'] = self.imageFileNameList[idx]

    def addColumns_diagnosis(self):
        old_var = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']
        new_var = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'OTHER']
        self.columnNames.extend(new_var)
        self.columnNames.append('ADD')
        targetTable = '../../derived_tables/OASIS/unique_mri_diag_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            id = row['ID']
            if id in self.content:
                for n_var, o_var in zip(new_var, old_var):
                    self.content[id][n_var] = row[o_var]
                if row['DE'] == '1' and row['AD'] == '1':
                    self.content[id]['ADD'] = 1
                elif row['DE'] == '1':
                    self.content[id]['ADD'] = 0

    def addColumns_demograph(self):
        self.columnNames.extend(['age', 'gender', 'education', 'race', 'hispanic', 'apoe'])
        targetTable = '../../raw_tables/OASIS/ADRC_Clinical_Data.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            id = row['Subject']
            if id in self.content:
                self.content[id]['age'] = int(self.content[id]['filename'].split('_')[1].strip('ses-d')) / 360.0 + float(row['ageAtEntry'])
                self.content[id]['age'] = self.content[id]['age'] // 1
                self.content[id]['apoe'] = row['apoe'].count('4') # new version, check with PJ
        targetTable = '../../raw_tables/OASIS/sub_demo.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            id = row['Subject']
            if id in self.content:
                gender = ''
                if row['SEX'] == '1': gender = 'male'
                if row['SEX'] == '2': gender = 'female'
                self.content[id]['gender'] = gender
                race = ''
                if row['RACE'] == '1': race = 'whi'
                if row['RACE'] == '2': race = 'blk'
                if row['RACE'] == '3': race = 'ind'
                if row['RACE'] == '4': race = 'haw'
                if row['RACE'] == '5': race = 'ans'
                if row['RACE'] == '6': race = 'mix'
                self.content[id]['race'] = race
                self.content[id]['education'] = row['Education']

    def addColumns_mmse(self):
        self.columnNames.extend(['mmse', 'gds', 'lm_imm', 'lm_del', 'digitF', 'digitFL', 'digitB', 'digitBL', 'animal', 'boston', 'trailA', 'trailB'])
        targetTable = '../../derived_tables/OASIS/mri_adrc_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        id_day_mmse = defaultdict(dict)
        id_day_gds  = defaultdict(dict)
        id_day_lm = defaultdict(dict)
        id_day_digit = defaultdict(dict)
        id_day_other = defaultdict(dict)
        for row in targetTable:
            id_day = row['folderName'].split('/')[-2]
            id, day = id_day.split('_')[0], id_day.split('_')[-1]
            id_day_mmse[id][day] = row['mmse']
            id_day_gds[id][day] = row['gds']
            id_day_lm[id][day] = (row['lm_imm'], row['lm_del'])
            id_day_digit[id][day] = (row['digitF'], row['digitFL'], row['digitB'], row['digitBL'])
            id_day_other[id][day] = (row['animal'], row['boston'], row['trailA'], row['trailB'])
        for id in self.content:
            if id in id_day_mmse:
                line = self.content[id]['filename']
                day = line.split('_')[1].strip('ses-')
                if day in id_day_mmse[id]:
                    if id_day_mmse[id][day] and 0 <= int(id_day_mmse[id][day]) <= 30:
                        self.content[id]['mmse'] = id_day_mmse[id][day]
                if day in id_day_gds[id]:
                    if id_day_gds[id][day] and 0 <= int(id_day_gds[id][day]) <= 15:
                        self.content[id]['gds'] = id_day_gds[id][day]
                if day in id_day_lm[id]:
                    if id_day_lm[id][day][0] and 0 <= int(id_day_lm[id][day][0]) <= 25:
                        self.content[id]['lm_imm'] = id_day_lm[id][day][0]
                    if id_day_lm[id][day][1] and 0 <= int(id_day_lm[id][day][1]) <= 25:
                        self.content[id]['lm_del'] = id_day_lm[id][day][1]
                if day in id_day_digit[id]:
                    if id_day_digit[id][day][0] and 0 <= int(id_day_digit[id][day][0]) <= 12:
                        self.content[id]['digitF'] = id_day_digit[id][day][0]
                    if id_day_digit[id][day][1] and 0 <= int(id_day_digit[id][day][1]) <= 8:
                        self.content[id]['digitFL'] = id_day_digit[id][day][1]
                    if id_day_digit[id][day][2] and 0 <= int(id_day_digit[id][day][2]) <= 12:
                        self.content[id]['digitB'] = id_day_digit[id][day][2]
                    if id_day_digit[id][day][3] and 0 <= int(id_day_digit[id][day][3]) <= 8:
                        self.content[id]['digitBL'] = id_day_digit[id][day][3]
                if day in id_day_other[id]:
                    if id_day_other[id][day][0] and 0 <= int(id_day_other[id][day][0]) <= 77:
                        self.content[id]['animal'] = id_day_other[id][day][0]
                    if id_day_other[id][day][1] and 0 <= int(id_day_other[id][day][1]) <= 30:
                        self.content[id]['boston'] = id_day_other[id][day][1]
                    if id_day_other[id][day][2] and 0 <= int(id_day_other[id][day][2]) <= 150:
                        self.content[id]['trailA'] = id_day_other[id][day][2]
                    if id_day_other[id][day][3] and 0 <= int(id_day_other[id][day][3]) <= 300:
                        self.content[id]['trailB'] = id_day_other[id][day][3]

    def addColumns_cdr(self):
        self.columnNames.extend(['cdr', 'cdrSum'])
        targetTable = '../../derived_tables/OASIS/mri_adrc_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        id_day_cdr = defaultdict(dict)
        for row in targetTable:
            id_day = row['folderName'].split('/')[-2]
            id, day = id_day.split('_')[0], id_day.split('_')[-1]
            id_day_cdr[id][day] = (row['cdr'], row['cdrSum'])
        for id in self.content:
            if id in id_day_cdr:
                line = self.content[id]['filename']
                day = line.split('_')[1].strip('ses-')
                if day in id_day_cdr[id]:
                    if id_day_cdr[id][day][0] and -0.1 < float(id_day_cdr[id][day][0]) < 3.1:
                        self.content[id]['cdr'] = id_day_cdr[id][day][0]
                    if id_day_cdr[id][day][1] and -0.1 < float(id_day_cdr[id][day][1]) < 18.1:
                        self.content[id]['cdrSum'] = id_day_cdr[id][day][1]

    def addColumns_NPIQ(self):
        vari_list = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                     'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN',
                     'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP']
        key_list = [a.split('_')[1] for a in vari_list]
        self.columnNames.extend(vari_list)
        targetTable = '../../derived_tables/OASIS/mri_adrc_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        id_day_npiq = defaultdict(dict)
        for row in targetTable:
            id_day = row['folderName'].split('/')[-2]
            id, day = id_day.split('_')[0], id_day.split('_')[-1]
            id_day_npiq[id][day] = {key : row[key] for key in key_list}
            for key in key_list:
                id_day_npiq[id][day][key+'SEV'] = row[key+'SEV']
        for id in self.content:
            if id in id_day_npiq:
                line = self.content[id]['filename']
                day = line.split('_')[1].strip('ses-')
                if day in id_day_npiq[id]:
                    for key in key_list:
                        if id_day_npiq[id][day][key] == '0':
                            self.content[id]['npiq_'+key] = '0'
                        elif id_day_npiq[id][day][key] == '1':
                            self.content[id]['npiq_' + key] = id_day_npiq[id][day][key+'SEV']

    def addColumns_FAQ(self):
        vari_list = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                     'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
        key_list = [a.split('_')[1] for a in vari_list]
        self.columnNames.extend(vari_list)
        targetTable = '../../derived_tables/OASIS/mri_adrc_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        id_day_faq = defaultdict(dict)
        for row in targetTable:
            id_day = row['folderName'].split('/')[-2]
            id, day = id_day.split('_')[0], id_day.split('_')[-1]
            id_day_faq[id][day] = {key: row[key] for key in key_list}
        for id in self.content:
            if id in id_day_faq:
                line = self.content[id]['filename']
                day = line.split('_')[1].strip('ses-')
                if day in id_day_faq[id]:
                    for key in key_list:
                        if id_day_faq[id][day][key] in ['0', '1', '2', '3']:
                            self.content[id]['faq_' + key] = id_day_faq[id][day][key]

    def addColumns_his(self):
        key_list = ['CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA',
                    'SEIZURES', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF',
                    'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR', 'PSYCDIS', 'ALCOHOL', 'TOBAC100',
                    'SMOKYRS', 'PACKSPER', 'ABUSOTHR']
        vari_list = ['his_' + a for a in key_list]
        self.columnNames.extend(vari_list)
        targetTable = '../../derived_tables/OASIS/mri_adrc_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        id_day_his = defaultdict(dict)
        for row in targetTable:
            id_day = row['folderName'].split('/')[-2]
            id, day = id_day.split('_')[0], id_day.split('_')[-1]
            id_day_his[id][day] = {key: row[key] for key in key_list}
        for id in self.content:
            if id in id_day_his:
                line = self.content[id]['filename']
                day = line.split('_')[1].strip('ses-')
                if day in id_day_his[id]:
                    for key in key_list:
                        if id_day_his[id][day][key] in ['9', '-4', '88', '99', '8']: continue
                        self.content[id]['his_' + key] = id_day_his[id][day][key]

    def combine_history_variables(self):
        self.columnNames.append('his_TBI')
        for id in self.content:
            row = self.content[id]
            try:
                # PMHx - TBI
                if '1' in (row['his_TRAUMBRF'], row['his_TRAUMEXT'], row['his_TRAUMCHR']):
                    self.content[id]['his_TBI'] = '1'
                elif '2' in (row['his_TRAUMBRF'], row['his_TRAUMEXT'], row['his_TRAUMCHR']):
                    self.content[id]['his_TBI'] = '2'
                elif row['his_TRAUMBRF']=='0' and row['his_TRAUMEXT']=='0' and row['his_TRAUMCHR']=='0':
                    self.content[id]['his_TBI'] = '0'
            except KeyError:
                pass

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_diagnosis()
        self.addColumns_demograph()
        self.addColumns_mmse()
        self.addColumns_cdr()
        self.addColumns_NPIQ()
        self.addColumns_FAQ()
        self.addColumns_his()
        self.combine_history_variables()
        with open(self.datasetName + '.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columnNames)
            writer.writeheader()
            for rid in self.content:
                writer.writerow(self.content[rid])

    def readcsv(self, csv_file):
        csvfile = open(csv_file, 'r')
        return csv.DictReader(csvfile)

if __name__ == "__main__":
    obj = TableData()
    obj.writeTable()