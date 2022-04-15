import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'AIBL'
        self.imageDir = '/data/datasets/AIBL/'
        self.imageFileNameList, self.RIDList, self.imgIDList = self.get_filenames_and_IDs(self.imageDir)
        self.columnNames = []
        self.content = defaultdict(dict) # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}
        self.time_table = self.get_time_table()

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.nii')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        imgIDList = [filename.split('_')[-1].strip('.nii')[1:] for filename in fileNameList]
        RIDList = [filename.split('_')[1] for filename in fileNameList]
        return fileNameList, RIDList, imgIDList

    def addColumns_path_filename(self):
        self.columnNames.extend(['path', 'filename', 'visit'])
        for idx, rid in enumerate(self.RIDList):
            self.content[rid]['path'] = self.imageDir
            self.content[rid]['filename'] = self.imageFileNameList[idx].replace('.nii', '.npy')
            self.content[rid]['visit'] = self.time_table[self.imgIDList[idx]]

    def get_time_table(self):
        convert = {'1': 'bl', '2': 'm18', '3': 'm36', '4': 'm54'}
        time_table = {} # the time_table maps image ID to visit code
        with open('../../raw_tables/AIBL/AIBL_4_27_2020.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time_table[row['Image Data ID']] = convert[row['Visit']]
        return time_table

    def addColumns_diagnosis(self):
        variables = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/AIBL/aibl_pdxconv_01-Jun-2018.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['RID'] in self.content and row['VISCODE'] == self.content[row['RID']]['visit']:
                if row['DXCURREN'] == '1':# NL healthy control
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['RID']][var] = 0   # 0 means no
                    self.content[row['RID']]['NC'] = 1      # 1 means yes
                    self.content[row['RID']]['COG'] = 0
                elif row['DXCURREN'] == '2':# MCI patient
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['RID']][var] = 0
                    self.content[row['RID']]['MCI'] = 1
                    self.content[row['RID']]['COG'] = 1
                elif row['DXCURREN'] == '3':
                    self.content[row['RID']]['COG'] = 2
                    self.content[row['RID']]['ADD'] = 1
                    for var in ['NC', 'MCI']:
                        self.content[row['RID']][var] = 0
                    for var in ['AD', 'DE']:
                        self.content[row['RID']][var] = 1

    def addColumns_mmse(self):
        self.columnNames.extend(['mmse'])
        targetTable = '../../raw_tables/AIBL/aibl_mmse_01-Jun-2018.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['RID'] in self.content and row['VISCODE'] == self.content[row['RID']]['visit']:
                self.content[row['RID']]['mmse'] = row['MMSCORE']

    def addColumns_cdr(self):
        self.columnNames.extend(['cdr'])
        targetTable = '../../raw_tables/AIBL/aibl_cdr_01-Jun-2018.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['RID'] in self.content and row['VISCODE'] == self.content[row['RID']]['visit']:
                self.content[row['RID']]['cdr'] = row['CDGLOBAL']

    def addColumns_lm(self):
        self.columnNames.extend(['lm_imm', 'lm_del'])
        targetTable = '../../raw_tables/AIBL/aibl_neurobat_01-Jun-2018.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['RID'] in self.content and row['VISCODE'] == self.content[row['RID']]['visit']:
                self.content[row['RID']]['lm_imm'] = row['LIMMTOTAL']
                self.content[row['RID']]['lm_del'] = row['LDELTOTAL']

    def addColumns_demograph(self):
        self.columnNames.extend(['age', 'gender'])
        targetTable = '../../raw_tables/AIBL/AIBL_4_27_2020.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Subject'] in self.content:
                self.content[row['Subject']]['age'] = int(row['Age'])
                self.content[row['Subject']]['gender'] = 'male' if row['Sex']== 'M' else 'female'

    def addColumns_apoe(self):
        self.columnNames.extend(['apoe'])
        targetTable = '../../raw_tables/AIBL/aibl_apoeres_01-Jun-2018.csv'  # to get age and gender which table you need to look into
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['RID'] in self.content:
                if (row['APGEN1'], row['APGEN2']) in [('3', '4'), ('4', '4')]:
                    self.content[row['RID']]['apoe'] = 1
                else:
                    self.content[row['RID']]['apoe'] = 0

    def addColumns_tesla(self):
        T15Table = self.readcsv('../../raw_tables/AIBL/aibl_mrimeta_01-Jun-2018.csv')
        T3Table = self.readcsv('../../raw_tables/AIBL/aibl_mri3meta_01-Jun-2018.csv')
        self.columnNames.extend(['Tesla'])
        self.content['135']['Tesla'] = 3.0 # manually checked
        self.content['448']['Tesla'] = 1.5  # manually checked
        self.content['653']['Tesla'] = 1.5  # manually checked
        self.content['340']['Tesla'] = 1.5  # manually checked
        for row in T15Table:
            if row['RID'] in self.content and row['VISCODE'] == self.content[row['RID']]['visit']:
                self.content[row['RID']]['Tesla'] = 1.5
        for row in T3Table:
            if row['RID'] in self.content and row['VISCODE'] == self.content[row['RID']]['visit']:
                self.content[row['RID']]['Tesla'] = 3.0

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_demograph()
        self.addColumns_apoe()
        self.addColumns_diagnosis()
        self.addColumns_mmse()
        self.addColumns_cdr()
        self.addColumns_lm()
        self.addColumns_tesla()
        with open(self.datasetName+'.csv', 'w') as csvfile:
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
