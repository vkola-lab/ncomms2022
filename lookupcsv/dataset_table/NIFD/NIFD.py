import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'NIFD'
        self.imageDir = '/data/datasets/NIFD/'
        self.imageFileNameList, self.RIDList = self.get_filenames_and_IDs(self.imageDir)
        self.columnNames = []
        self.content = defaultdict(dict) # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.nii')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        RIDList = [filename[5:13] for filename in fileNameList]
        return fileNameList, RIDList

    def addColumns_path_filename(self):
        self.columnNames.extend(['path', 'filename'])
        for idx, id in enumerate(self.RIDList):
            self.content[id]['path'] = self.imageDir
            self.content[id]['filename'] = self.imageFileNameList[idx].replace('.nii', '.npy')

    def addColumns_demograph(self):
        self.columnNames.extend(['age', 'gender', 'education', 'race'])
        targetTable = '../../raw_tables/NIFD/NIFD_visit.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Subject'] in self.content:
                self.content[row['Subject']]['age'] = row['Age']
                self.content[row['Subject']]['gender'] = 'male' if row['Sex']=='M' else 'female'
        targetTable = '../../raw_tables/NIFD/NIFD_Clinical_Data_2017_final_updated.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['LONI_ID'] in self.content and row['VISIT_NUMBER'] == '1':
                self.content[row['LONI_ID']]['education'] = row['EDUCATION']
                race = ''
                if row['RACE'] == '1': race = 'whi'
                elif row['RACE'] == '2': race = 'blk'
                elif row['RACE'] == '3': race = 'ans'
                elif row['RACE'] == '4': race = 'haw'
                elif row['RACE'] == '5': race = 'mix'
                self.content[row['LONI_ID']]['race'] = race

    def addColumns_diagnosis(self):
        variables = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/NIFD/NIFD_Clinical_Data_2017_final_updated.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['LONI_ID'] in self.content and row['VISIT_NUMBER']=='1':
                if row['DX'] in ['SV', 'BV', 'PNFA']: # FTD
                    for var in ['NC', 'MCI']: # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['LONI_ID']][var] = 0
                    self.content[row['LONI_ID']]['FTD'] = 1
                    self.content[row['LONI_ID']]['DE'] = 1
                    self.content[row['LONI_ID']]['COG'] = 2
                    self.content[row['LONI_ID']]['ADD'] = 0
                elif row['DX'] == 'CON':  # healthy control
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['LONI_ID']][var] = 0   # 0 means no
                    self.content[row['LONI_ID']]['NC'] = 1      # 1 means yes
                    self.content[row['LONI_ID']]['COG'] = 0
                else:
                    print(row['DX']) # PATIENT; L_SD, Patient (Others)

    def addColumns_mmse_cdr_moca(self):
        self.columnNames.extend(['mmse', 'cdr', 'cdrSum', 'moca'])
        targetTable = '../../raw_tables/NIFD/NIFD_Clinical_Data_2017_final_updated.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['LONI_ID'] in self.content and row['VISIT_NUMBER'] == '1':
                self.content[row['LONI_ID']]['mmse'] = row['MMSE_TOT'] if row['MMSE_TOT'] and 0<=int(row['MMSE_TOT'])<=30 else ''
                self.content[row['LONI_ID']]['cdr'] = row['CDR_TOT'] if row['CDR_TOT'] and -0.1<float(row['CDR_TOT'])<3.1 else ''
                self.content[row['LONI_ID']]['cdrSum'] = row['CDR_BOX_SCORE'] if row['CDR_BOX_SCORE'] and -0.1<=float(row['CDR_BOX_SCORE'])<18.1 else ''
                self.content[row['LONI_ID']]['moca'] = row['MOCA_TotWithEduc'] if row['MOCA_TotWithEduc'] and 0<=int(row['MOCA_TotWithEduc'])<=30 else ''

    def addColumns_other(self):
        self.columnNames.extend(['digitFL', 'digitBL', 'animal', 'boston', 'gds', 'Dwords'])
        targetTable = '../../raw_tables/NIFD/NIFD_Clinical_Data_2017_final_updated.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['LONI_ID'] in self.content and row['VISIT_NUMBER'] == '1':
                if row['DIGITFW'] and 0 <= int(row['DIGITFW']):
                    self.content[row['LONI_ID']]['digitFL'] = row['DIGITFW']
                if row['DIGITBW'] and 0 <= int(row['DIGITBW']):
                    self.content[row['LONI_ID']]['digitBL'] = row['DIGITBW']
                if row['ANCORR'] and 0 <= int(row['ANCORR']) <= 77:
                    self.content[row['LONI_ID']]['animal'] = row['ANCORR']
                if row['BNTCORR'] and 0 <= int(row['BNTCORR']) <= 30:
                    self.content[row['LONI_ID']]['boston'] = row['BNTCORR']
                if row['GDS15TO'] and 0 <= int(row['GDS15TO']) <= 15:
                    self.content[row['LONI_ID']]['gds'] = row['GDS15TO']
                if row['DCORR'] and 0 <= int(row['DCORR']):
                    self.content[row['LONI_ID']]['Dwords'] = row['DCORR']


    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_demograph()
        self.addColumns_diagnosis()
        self.addColumns_mmse_cdr_moca()
        self.addColumns_other()
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
