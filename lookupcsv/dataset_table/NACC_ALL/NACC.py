import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'NACC'
        self.imageDir = '/data_2/NACC_ALL/npy/'
        self.content = defaultdict(dict)  # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}
        self.columnNames = []
        self.imageFileNameList, self.headerList = self.get_filenames_and_IDs(self.imageDir)
        self.ID_date_to_header, self.header_to_ID, self.ID_to_header = self.get_ID_date_to_zip_map()

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.npy')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        ImageHeaderList = [filename.split('_')[0] for filename in fileNameList]
        return fileNameList, ImageHeaderList

    def get_ID_date_to_zip_map(self):
        ID_date_to_zip = {}
        zip_to_NACCID = {}
        NACCID_to_zip = {}
        targetTable = '../../derived_tables/NACC/zip_id_data_6months.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            ID_date_to_zip[(row['ID'], row['Year'], row['Month'], row['Day'])] = row['zipname'].strip('.zip')
            zip_to_NACCID[row['zipname'].strip('.zip')] = row['ID']
            NACCID_to_zip[row['ID']] = row['zipname'].strip('.zip')
        targetTable = '../../derived_tables/NACC_ALL/zip_id_data_6months.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            ID_date_to_zip[(row['ID'], row['Year'], row['Month'], row['Day'])] = row['zipname'].strip('.zip').split('_')[0]
            zip_to_NACCID[row['zipname'].strip('.zip').split('_')[0]] = row['ID']
            NACCID_to_zip[row['ID']] = row['zipname'].strip('.zip').split('_')[0]
        return ID_date_to_zip, zip_to_NACCID, NACCID_to_zip

    def addColumns_path_filename(self):
        self.columnNames.extend(['path', 'filename', 'ID'])
        for idx, header in enumerate(self.headerList):
            id = self.header_to_ID[header]
            self.content[id]['path'] = self.imageDir
            self.content[id]['filename'] = self.imageFileNameList[idx]
            self.content[id]['ID'] = id

    def addColumns_based_on_ID(self):
        self.columnNames.extend(['gender', 'education', 'race', 'hispanic', 'apoe'])
        for targetTable in ['../../raw_tables/NACC/kolachalama06022020.csv',
                            '../../raw_tables/NACC_ALL/kolachalama12042020.csv']:
            targetTable = self.readcsv(targetTable)
            for row in targetTable:
                if row['NACCID'] in self.content:
                    id = row['NACCID']

                    gender = ''
                    if row['SEX'] == '1': gender = 'male'
                    if row['SEX'] == '2': gender = 'female'
                    self.special_add(id, 'gender', gender)

                    edu = row['EDUC'] if int(row['EDUC']) <= 36 else ''
                    self.special_add(id, 'education', edu)

                    race = ''
                    if row['RACE'] == '1': race = 'whi'
                    if row['RACE'] == '2': race = 'blk'
                    if row['RACE'] == '3': race = 'ind'
                    if row['RACE'] == '4': race = 'haw'
                    if row['RACE'] == '5': race = 'ans'
                    if row['RACE'] == '6': race = 'mix'
                    self.special_add(id, 'race', race)

                    hispanic = ''
                    if row['HISPANIC'] == '1': hispanic = 'yes'
                    if row['HISPANIC'] == '0': hispanic = 'no'
                    self.special_add(id, 'hispanic', hispanic)

                    if row['NACCNE4S'] in ['0', '1', '2']:  # '9' is not availabel
                        apoe = row['NACCNE4S']
                        self.special_add(id, 'apoe', apoe)


    def addColumns_history(self):
        variables = ['PRIMLANG', 'HANDED', 'LIVSIT', 'INDEPEND', 'RESIDENC', 'MARISTAT', 'INRELTO',
                    'NACCFAM', 'NACCNREX', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVPACDEF', 
                    'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'SEIZURES', 'TBI', 'TBIBRIEF', 'TBIEXTEN',
                     'TBIWOLOS', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'HYPERTEN', 'HYPERCHO',
                    'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR',
                     'PSYCDIS', 'NPSYDEV', 'OCD', 'ANXIETY', 'SCHIZ', 'BIPOLAR', 'PTSD', 'ALCOHOL',
                     'TOBAC100', 'SMOKYRS', 'PACKSPER', 'ABUSOTHR', 'CVOTHRX', 'PSYOTHRX', 'NCOTHR', 'NCOTHRX',
                     'ABUSX', 'DIABTYPE']
        self.columnNames.extend(['his_'+v for v in variables])
        for targetTable in ['../../raw_tables/NACC/kolachalama06022020.csv',
                            '../../raw_tables/NACC_ALL/kolachalama12042020.csv']:
            targetTable = self.readcsv(targetTable)
            for row in targetTable:
                if row['NACCID'] not in self.content: continue
                key = (row['NACCID'],row['VISITYR'],row['VISITMO'],row['VISITDAY'])
                if key in self.ID_date_to_header and self.ID_date_to_header[key] in self.headerList:
                    id = row['NACCID']
                    for vari in variables:
                        if vari not in row or row[vari] in ['9', '-4', '88', '99', '8']: row[vari] = ''
                        self.special_add(id, 'his_'+vari, row[vari])

    def combine_history_variables(self):
        for id in self.content:
            row = self.content[id]
            try:
                # PMHx - Bypass
                if not row['his_CVPACE']: self.content[id]['his_CVPACE'] = row['his_CVPACDEF']
            except KeyError:
                pass
            try:
                # PMHx - TBI
                if '1' in (row['his_TBI'], row['his_TRAUMBRF'], row['his_TRAUMEXT'], row['his_TRAUMCHR']):
                    self.content[id]['his_TBI'] = '1'
                elif '2' in (row['his_TBI'], row['his_TRAUMBRF'], row['his_TRAUMEXT'], row['his_TRAUMCHR']):
                    self.content[id]['his_TBI'] = '2'
                elif row['his_TRAUMBRF']=='0' and row['his_TRAUMEXT']=='0' and row['his_TRAUMCHR']=='0':
                    self.content[id]['his_TBI'] = '0'
            except KeyError:
                pass
            try:
                # PMHx - Psych
                if '1' in (row['his_PSYCDIS'], row['his_NPSYDEV'], row['his_OCD'], row['his_ANXIETY'],
                           row['his_SCHIZ'], row['his_BIPOLAR'], row['his_PTSD']):
                    self.content[id]['his_PSYCDIS'] = '1'
                elif '2' in (row['his_PSYCDIS'], row['his_NPSYDEV'], row['his_OCD'], row['his_ANXIETY'],
                           row['his_SCHIZ'], row['his_BIPOLAR'], row['his_PTSD']):
                    self.content[id]['his_PSYCDIS'] = '2'
                elif row['his_NPSYDEV']=='0' and row['his_OCD']=='0' and row['his_ANXIETY']=='0' and \
                     row['his_SCHIZ']=='0' and row['his_BIPOLAR']=='0' and row['his_PTSD']=='0':
                    self.content[id]['his_PSYCDIS'] = '0'
            except KeyError:
                pass

    def special_add(self, id, variable, fill_val):
        if variable not in self.content[id] or not self.content[id][variable]:
            self.content[id][variable] = fill_val

    def addColumns_based_on_ID_date(self):
        self.columnNames.extend(['age', 'mmse', 'moca', 'cdr', 'cdrSum', 'boston', 'digitB', 'digitBL', 'digitF', 'digitFL',
                                 'animal', 'Fwords', 'gds', 'lm_imm', 'lm_del', 'lm_memtime', 'craft_imm', 'craft_del', 'mint', 'numberB',
                                 'numberBL', 'numberF', 'numberFL', 'trailA', 'trailB'])
        for targetTable in ['../../raw_tables/NACC/kolachalama06022020.csv', '../../raw_tables/NACC_ALL/kolachalama12042020.csv']:
            targetTable = self.readcsv(targetTable)
            for row in targetTable:
                if row['NACCID'] not in self.content: continue
                key = (row['NACCID'],row['VISITYR'],row['VISITMO'],row['VISITDAY'])
                if key in self.ID_date_to_header and self.ID_date_to_header[key] in self.headerList:
                    id = row['NACCID']
                    self.special_add(id, 'age', row['NACCAGE'])
                    if row['NACCMMSE'] and 0 <= int(row['NACCMMSE']) <= 30:
                        self.special_add(id, 'mmse', row['NACCMMSE'])
                    if row['NACCMOCA'] and 0 <= int(row['NACCMOCA']) <= 30:
                        self.special_add(id, 'moca', row['NACCMOCA'])
                    if row['CDRGLOB']:
                        self.special_add(id, 'cdr', row['CDRGLOB'])
                        self.special_add(id, 'cdrSum', row['CDRSUM'])
                    if row['BOSTON'] and 0 <= int(row['BOSTON']) <= 30:
                        self.special_add(id, 'boston', row['BOSTON'])
                    if row['DIGIBLEN'] and 0 <= int(row['DIGIBLEN']) <= 8:
                        self.special_add(id, 'digitBL', row['DIGIBLEN'])
                    if row['DIGIFLEN'] and 0 <= int(row['DIGIFLEN']) <= 8:
                        self.special_add(id, 'digitFL', row['DIGIFLEN'])
                    if row['DIGIB'] and 0 <= int(row['DIGIB']) <= 12:
                        self.special_add(id, 'digitB', row['DIGIB'])
                    if row['DIGIF'] and 0 <= int(row['DIGIF']) <= 12:
                        self.special_add(id, 'digitF', row['DIGIF'])
                    if row['ANIMALS'] and 0 <= int(row['ANIMALS']) <= 77:
                        self.special_add(id, 'animal', row['ANIMALS'])
                    if row['UDSVERFC'] and 0 <= int(row['UDSVERFC']) <= 40:
                        self.special_add(id, 'Fwords', row['UDSVERFC'])
                    if row['NACCGDS'] and 0 <= int(row['NACCGDS']) <=15:
                        self.special_add(id, 'gds', row['NACCGDS'])
                    if row['LOGIMEM'] and 0 <= int(row['LOGIMEM']) <= 25:
                        self.special_add(id, 'lm_imm', row['LOGIMEM'])
                    if row['MEMUNITS'] and 0 <= int(row['MEMUNITS']) <= 25:
                        self.special_add(id, 'lm_del', row['MEMUNITS'])

                    if row['MEMTIME'] and 0 <= int(row['MEMTIME']) <= 85:
                        self.special_add(id, 'lm_memtime', row['MEMTIME'])

                    if row['CRAFTURS'] and 0 <= int(row['CRAFTURS']) <= 25:
                        self.special_add(id, 'craft_imm', row['CRAFTURS'])
                    if row['CRAFTDRE'] and 0 <= int(row['CRAFTDRE']) <= 25:
                        self.special_add(id, 'craft_del', row['CRAFTDRE'])
                    if row['MINTTOTS'] and 0 <= int(row['MINTTOTS']) <= 32:
                        self.special_add(id, 'mint', row['MINTTOTS'])
                    if row['DIGFORCT'] and 0 <= int(row['DIGFORCT']) <= 14:
                        self.special_add(id, 'numberF', row['DIGFORCT'])
                    if row['DIGBACCT'] and 0 <= int(row['DIGBACCT']) <= 14:
                        self.special_add(id, 'numberB', row['DIGBACCT'])
                    if row['DIGFORSL'] and 0 <= int(row['DIGFORSL']) <= 9:
                        self.special_add(id, 'numberFL', row['DIGFORSL'])
                    if row['DIGBACLS'] and 0 <= int(row['DIGBACLS']) <= 8:
                        self.special_add(id, 'numberBL', row['DIGBACLS'])
                    if row['TRAILA'] and 0 <= int(row['TRAILA']) <= 150:
                        self.special_add(id, 'trailA', row['TRAILA'])
                    if row['TRAILB'] and 0 <= int(row['TRAILB']) <= 300:
                        self.special_add(id, 'trailB', row['TRAILB'])


    def addColumns_NPIQ(self):
        vari_list = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                     'npiq_ANX', 'npiq_ELAT', 'npiq_APA',  'npiq_DISN',
                     'npiq_IRR', 'npiq_MOT',  'npiq_NITE', 'npiq_APP']
        self.columnNames.extend(vari_list)
        for targetTable in ['../../raw_tables/NACC/kolachalama06022020.csv',
                            '../../raw_tables/NACC_ALL/kolachalama12042020.csv']:
            targetTable = self.readcsv(targetTable)
            for row in targetTable:
                if row['NACCID'] not in self.content: continue
                key = (row['NACCID'], row['VISITYR'], row['VISITMO'], row['VISITDAY'])
                if key in self.ID_date_to_header and self.ID_date_to_header[key] in self.headerList:
                    id = row['NACCID']
                    for vari in vari_list:
                        name = vari.split('_')[1]
                        if row[name] and row[name] == '0':
                            self.content[id][vari] = '0'
                        elif row[name] == '1':
                            if row[name+'SEV'] and row[name+'SEV'] in ['1', '2', '3']:
                                self.content[id][vari] = row[name+'SEV']

    def addColumns_FAQ(self):
        vari_list = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                     'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
        self.columnNames.extend(vari_list)
        for targetTable in ['../../raw_tables/NACC/kolachalama06022020.csv',
                            '../../raw_tables/NACC_ALL/kolachalama12042020.csv']:
            targetTable = self.readcsv(targetTable)
            for row in targetTable:
                if row['NACCID'] not in self.content: continue
                key = (row['NACCID'], row['VISITYR'], row['VISITMO'], row['VISITDAY'])
                if key in self.ID_date_to_header and self.ID_date_to_header[key] in self.headerList:
                    id = row['NACCID']
                    for vari in vari_list:
                        name = vari.split('_')[1]
                        if name in row and row[name] in ['0', '1', '2', '3']: # if want to know whether the info is missing or N.A. add '8', '9'
                            self.content[id][vari] = row[name]


    def addColumns_diagnosis(self):
        old_var = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']
        new_var = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'OTHER']
        self.columnNames.extend(new_var)
        self.columnNames.append('ADD')
        targetTable = '../../derived_tables/NACC/unique_mri_diag_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            header = row['zipname'].strip('.zip')
            id = self.header_to_ID[header]
            if id in self.content:
                for n_var, o_var in zip(new_var, old_var):
                    self.content[id][n_var] = row[o_var]
                if row['DE'] == '1' and row['AD'] == '1':
                    self.content[id]['ADD'] = 1
                elif row['DE'] == '1':
                    self.content[id]['ADD'] = 0
        targetTable = '../../derived_tables/NACC_ALL/unique_mri_diag_table_6months.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            header = row['zipname'].strip('.zip').split('_')[0]
            id = self.header_to_ID[header]
            if id in self.content:
                for n_var, o_var in zip(new_var, old_var):
                    self.content[id][n_var] = row[o_var]
                if row['DE'] == '1' and row['AD'] == '1':
                    self.content[id]['ADD'] = 1
                elif row['DE'] == '1':
                    self.content[id]['ADD'] = 0

    def convert_missing(self):
        MINT_Boston = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:6, 7:7, 8:7, 9:8, 10:8, 11:9, 12:9, 13:10, 14:11, 15:11, 16:12, 17:13,
                       18:14, 19:15, 20:16, 21:17, 22:18, 23:20, 24:21, 25:22, 26:24, 27:25, 28:26, 29:27, 30:28, 31:29, 32:30}
        craft_lm_imm = {0:0, 1:0, 2:1, 3:2, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8, 11:9, 12:10, 13:11,
                        14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 24:22, 25:23}
        craft_lm_del = {0:0, 1:1, 2:3, 3:3, 4:4, 5:5, 6:5, 7:6, 8:7, 9:8, 10:8, 11:9, 12:10, 13:11,
                        14:12, 15:13, 16:14, 17:15, 18:16, 19:18, 20:19, 21:20, 22:21, 23:22, 24:23, 25:24}
        number_digit_BL = {0:0, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:7}
        number_digit_FL = {0:0, 3:4, 4:5, 5:5, 6:6, 7:7, 8:8, 9:8}
        number_digit_B = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:12, 14:12}
        number_digit_F = {0:0, 1:1, 2:2, 3:3, 4:5, 5:6, 6:7, 7:8, 8:9, 9:9, 10:10, 11:11, 12:11, 13:12, 14:12}
        moca_mmse = {0:3, 1:6, 2:8, 3:9, 4:10, 5:11, 6:12, 7:13, 8:14, 9:15, 10:16, 11:17, 12:19, 13:20, 14:21, 15:22, 16:22,
                     17:23, 18:24, 19:25, 20:26, 21:26, 22:27, 23:28, 24:28, 25:29, 26:29, 27:29, 28:30, 29:30, 30:30}
        for rid in self.content:
            case = self.content[rid]
            if 'boston' not in case and 'mint' in case and case['mint']:
                self.content[rid]['boston'] = MINT_Boston[int(case['mint'])]
            if 'lm_imm' not in case and 'craft_imm' in case and case['craft_imm']:
                self.content[rid]['lm_imm'] = craft_lm_imm[int(case['craft_imm'])]
            if 'lm_del' not in case and 'craft_del' in case and case['craft_del']:
                self.content[rid]['lm_del'] = craft_lm_del[int(case['craft_del'])]
            if 'digitBL' not in case and 'numberBL' in case and case['numberBL']:
                self.content[rid]['digitBL'] = number_digit_BL[int(case['numberBL'])]
            if 'digitFL' not in case and 'numberFL' in case and case['numberFL']:
                self.content[rid]['digitFL'] = number_digit_FL[int(case['numberFL'])]
            if 'digitB' not in case and 'numberB' in case and case['numberB']:
                self.content[rid]['digitB'] = number_digit_B[int(case['numberB'])]
            if 'digitF' not in case and 'numberF' in case and case['numberF']:
                self.content[rid]['digitF'] = number_digit_F[int(case['numberF'])]
            if 'mmse' not in case and 'moca' in case and case['moca']:
                self.content[rid]['mmse'] = moca_mmse[int(case['moca'])]

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_diagnosis()
        self.addColumns_based_on_ID()
        self.addColumns_based_on_ID_date()
        # self.addColumns_csf()
        self.addColumns_NPIQ()
        self.addColumns_FAQ()
        self.addColumns_history()
        self.convert_missing()
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



