import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'PPMI'
        self.imageDir = '/data/datasets/PPMI/'
        self.imageFileNameList, self.RIDList = self.get_filenames_and_IDs(self.imageDir)
        self.columnNames = []
        self.content = defaultdict(dict) # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.npy')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        RIDList = [filename.split('_')[1] for filename in fileNameList]
        return fileNameList, RIDList

    def addColumns_eventID(self):
        self.columnNames.extend(['EVENT_ID'])
        targetTable = '../../derived_tables/PPMI/PPMI_scans_time.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            id = row['filename'].split('_')[1]
            if id in self.content and row['filename'].replace('.nii', '.npy')==self.content[id]['filename']:
                self.content[id]['EVENT_ID'] = row['EVENT_ID']

    def addColumns_path_filename(self):
        self.columnNames.extend(['path', 'filename'])
        for idx, id in enumerate(self.RIDList):
            self.content[id]['path'] = self.imageDir
            self.content[id]['filename'] = self.imageFileNameList[idx]

    def addColumns_demograph(self):
        imageID_to_age, ID_to_gender = {}, {}
        with open('../../raw_tables/PPMI/PPMI_5_28_2020.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                imageID_to_age[row['Image Data ID']] = row['Age']
                ID_to_gender[row['Subject']] = 'male' if row['Sex']=='M' else 'female'
        self.columnNames.extend(['age', 'gender', 'education', 'hispanic', 'race'])
        targetTable = '../../raw_tables/PPMI/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PATNO'] in self.content:
                self.content[row['PATNO']]['education'] = row['EDUCYRS']
                self.content[row['PATNO']]['hispanic'] = row['HISPLAT']
        for id in self.content:
            imageID = self.content[id]['filename'].split('_')[-1][1:-4]
            self.content[id]['age'] = imageID_to_age[imageID]
            self.content[id]['gender'] = ID_to_gender[id]

    def addColumns_race(self):
        targetTable = '../../raw_tables/PPMI/Screening___Demographics.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PATNO'] in self.content:
                race = [row['RAINDALS'], row['RAASIAN'], row['RABLACK'], row['RAHAWOPI'], row['RAWHITE']]
                if race.count('1') > 1:
                    self.content[row['PATNO']]['race'] = 'mix'
                elif row['RAINDALS'] == '1':
                    self.content[row['PATNO']]['race'] = 'ind'
                elif row['RAASIAN'] == '1':
                    self.content[row['PATNO']]['race'] = 'ans'
                elif row['RABLACK'] == '1':
                    self.content[row['PATNO']]['race'] = 'blk'
                elif row['RAHAWOPI'] == '1':
                    self.content[row['PATNO']]['race'] = 'haw'
                elif row['RAWHITE'] == '1':
                    self.content[row['PATNO']]['race'] = 'whi'

    def addColumns_diagnosis(self):
        variables = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']
        self.columnNames.extend(variables)
        #################### firstly fill PD label
        imageID_to_PD = {}
        with open('../../raw_tables/PPMI/PPMI_5_28_2020.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                imageID_to_PD[row['Image Data ID']] = row['Group']
        for id in self.content:
            imageID = self.content[id]['filename'].split('_')[-1][1:-4]
            if imageID_to_PD[imageID] == 'Control':
                self.content[id]['PD'] = 0
            elif imageID_to_PD[imageID] in ['PD', 'GenCohort PD', 'GenReg PD']:
                self.content[id]['PD'] = 1
        #################### then fill cognitive status and other possible types of dementia
        targetTable = '../../raw_tables/PPMI/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PATNO'] in self.content and row['EVENT_ID']==self.content[row['PATNO']]['EVENT_ID']:
                if row['cogstate']=='1': #NC
                    for var in ['MCI', 'DE', 'COG', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                        self.content[row['PATNO']][var] = 0
                    self.content[row['PATNO']]['NC'] = 1
                elif row['cogstate']=='2': #MCI
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                        self.content[row['PATNO']][var] = 0
                    self.content[row['PATNO']]['MCI'] = 1
                    self.content[row['PATNO']]['COG'] = 1
                elif row['cogstate']=='3': #Dementia
                    for var in ['NC', 'MCI']:
                        self.content[row['PATNO']][var] = 0
                    self.content[row['PATNO']]['DE'] = 1
                    self.content[row['PATNO']]['COG'] = 2
                    self.content[row['PATNO']]['PDD'] = 0
                    self.content[row['PATNO']]['ADD'] = 0
                    if row['primdiag'] == '1':
                        self.content[row['PATNO']]['PD'] = 1
                        self.content[row['PATNO']]['PDD'] = 1
                    if row['primdiag'] == '2':
                        self.content[row['PATNO']]['AD'] = 1
                        self.content[row['PATNO']]['ADD'] = 1
                    if row['primdiag'] == '3':
                        self.content[row['PATNO']]['FTD'] = 1
                    if row['primdiag'] == '5':
                        self.content[row['PATNO']]['DLB'] = 1
                    else:
                        self.content[row['PATNO']]['OTHER'] = 1


    def addColumns_moca(self):
        self.columnNames.extend(['moca'])
        targetTable = '../../raw_tables/PPMI/Montreal_Cognitive_Assessment__MoCA_.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PATNO'] in self.content:
                if row['EVENT_ID']==self.content[row['PATNO']]['EVENT_ID']:
                    self.content[row['PATNO']]['moca'] = row['MCATOT']
                elif row['EVENT_ID']=='SC' and self.content[row['PATNO']]['EVENT_ID']=='BL':
                    self.content[row['PATNO']]['moca'] = row['MCATOT']

    def addColumns_trailAB(self):
        self.columnNames.extend(['trailA', 'trailB'])
        targetTable = '../../raw_tables/PPMI/Trail_Making_A_and_B.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PATNO'] in self.content:
                if row['EVENT_ID'] == self.content[row['PATNO']]['EVENT_ID'] or (row['EVENT_ID']=='SC' and self.content[row['PATNO']]['EVENT_ID']=='BL'):
                    self.content[row['PATNO']]['trailA'] = row['TMTASEC']
                    self.content[row['PATNO']]['trailB'] = row['TMTBSEC']

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_eventID()
        self.addColumns_demograph()
        self.addColumns_race()
        self.addColumns_diagnosis()
        self.addColumns_moca()
        self.addColumns_trailAB()
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




#########################################
################### then fill cognitive status
# with open('../../raw_tables/PPMI/Cognitive_Categorization.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         if row['PATNO'] in self.content and row['EVENT_ID']==self.content[row['PATNO']]['EVENT_ID']:
#             if row['COGCAT']=='1': # NC
#                 for var in ['MCI', 'DE', 'COG', 'AD', 'FTD', 'VD', 'DLB', 'OTHER']:
#                     self.content[row['PATNO']][var] = 0
#                 self.content[row['PATNO']]['NC'] = 1
#             elif row['COGCAT']=='3': # MCI
#                 for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'OTHER']:
#                     self.content[row['PATNO']][var] = 0
#                 self.content[row['PATNO']]['MCI'] = 1
#                 self.content[row['PATNO']]['COG'] = 1
#             elif row['COGCAT']=='4': # Dementia
#                 for var in ['NC', 'MCI']:
#                     self.content[row['PATNO']][var] = 0
#                 self.content[row['PATNO']]['DE'] = 1
#                 self.content[row['PATNO']]['COG'] = 2
########################################
# if row['PATNO'] in self.content and row['EVENT_ID']==self.content[row['PATNO']]['EVENT_ID']:
#     if 'DE' in self.content[row['PATNO']] and self.content[row['PATNO']]['DE']=='1':
#         if row['primdiag'] == '1':
#             self.content[row['PATNO']]['PD'] = 1
#         if row['primdiag'] == '2':
#             self.content[row['PATNO']]['AD'] = 1
#         if row['primdiag'] == '3':
#             self.content[row['PATNO']]['FTD'] = 1
#         if row['primdiag'] == '5':
#             self.content[row['PATNO']]['DLB'] = 1
#         else:
#             self.content[row['PATNO']]['OTHER'] = 1