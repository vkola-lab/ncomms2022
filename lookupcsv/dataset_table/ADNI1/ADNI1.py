import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'ADNI1'
        self.imageDir = '/data/datasets/ADNI1/'
        self.imageFileNameList, self.RIDList = self.get_filenames_and_IDs(self.imageDir)
        self.columnNames = []
        self.content = defaultdict(dict) # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.nii')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        IDList = [filename[5:15] for filename in fileNameList]
        RIDList = [id[-4:].lstrip("0") for id in IDList]
        return fileNameList, RIDList

    def addColumns_path_filename(self):
        self.columnNames.extend(['path', 'filename'])
        for idx, id in enumerate(self.RIDList):
            self.content[id]['path'] = self.imageDir
            self.content[id]['filename'] = self.imageFileNameList[idx].replace('.nii', '.npy')

    def addColumns_demograph(self):
        self.columnNames.extend(['age', 'gender', 'education', 'hispanic', 'race'])
        targetTable = '../../raw_tables/ADNI/PTDEMOG.csv'  # to get age and gender which table you need to look into
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                self.content[row['RID']]['age'] = int(row['USERDATE'][:4])-int(row['PTDOBYY'])
                self.content[row['RID']]['gender'] = 'male' if row['PTGENDER']=='1' else 'female'
                self.content[row['RID']]['education'] = int(row['PTEDUCAT'])
                self.content[row['RID']]['hispanic'] = int(row['PTETHCAT'])
                race = ''
                if row['PTRACCAT'] == '1':
                    race = 'ind'
                elif row['PTRACCAT'] == '2':
                    race = 'ans'
                elif row['PTRACCAT'] == '3':
                    race = 'haw'
                elif row['PTRACCAT'] == '4':
                    race = 'blk'
                elif row['PTRACCAT'] == '5':
                    race = 'whi'
                elif row['PTRACCAT'] == '6':
                    race = 'mix'
                self.content[row['RID']]['race'] = race

    def addColumns_apoe(self):
        self.columnNames.extend(['apoe'])
        targetTable = '../../raw_tables/ADNI/APOERES.csv'  # to get age and gender which table you need to look into
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['RID'] in self.content:
                if (row['APGEN1'], row['APGEN2']) in [('3', '4'), ('4', '4')]:
                    self.content[row['RID']]['apoe'] = 1
                else:
                    self.content[row['RID']]['apoe'] = 0

    def addColumns_diagnosis(self):
        variables = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ADNI_DXSUM_PDXCONV.csv'
        exclusTable = '../../raw_tables/ADNI/EXCLUSIO.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                if row['DXCURREN'] == '1': # NL healthy control
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['RID']][var] = 0   # 0 means no
                    self.content[row['RID']]['NC'] = 1      # 1 means yes
                    self.content[row['RID']]['COG'] = 0
                elif row['DXCURREN'] == '2': # MCI patient
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['RID']][var] = 0
                    self.content[row['RID']]['MCI'] = 1
                    self.content[row['RID']]['COG'] = 1
                elif row['DXCURREN'] == '3': # Dementia patient
                    self.content[row['RID']]['COG'] = 2
                    self.content[row['RID']]['ADD'] = 1
                    for var in ['NC', 'MCI']:
                        self.content[row['RID']][var] = 0
                    for var in ['AD', 'DE']:
                        self.content[row['RID']][var] = 1
                    if row['DXOTHDEM'] != '-4': # all AD cases has DXOTHDEM = -4, thus other dementia info is unknown
                        print('found AD case with other dementia info')
                else:
                    print(row['DXCURREN']) # no print out, DXCURREN can only take value 1, 2, 3
                if row['DXPARK'] != '-4':
                    print('found a case with PD info')  # no print here, turns out all DXPARK=-4
                    self.content[row['RID']]['PD'] = row['DXPARK']
        exclusTable = self.readcsv(exclusTable)
        for row in exclusTable: # this table is only for ADNI1 screening case, use the table to check whether other neurologic diseases exist
            if row['VISCODE'] == 'sc' and row['RID'] in self.content:
                if row['EXNEURO'] == '0': # no other neurologic disease
                    self.content[row['RID']]['PD'] = 0  # including no parkinson
                    self.content[row['RID']]['VD'] = 0
                    self.content[row['RID']]['PDD'] = 0
                elif row['EXNEURO'] == '1': # there is other neurologic disease
                    # check subtype using DXODES in target table
                    print('found other neruologic diseases')
                    print(self.content[row['RID']])

    def addColumns_mmse(self):
        # variables = ['mmse', 'mmse_orient', 'mmse_workMem', 'mmse_concen', 'mmse_memRec', 'mmse_lang', 'mmse_visuSpa']
        # old_variables = ['mmse','Orientation','Working Memory','Concentration','Memory Recall','Language','Visuospatial']
        variables = ['mmse']
        old_variables = ['mmse']
        self.columnNames.extend(variables)
        targetTable = '../../derived_tables/ADNI/ADNI_MMSE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_cdr(self):
        variables = ['cdr', 'cdrSum']
        old_variables = ['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/CDR.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                sumScore = 0
                for var in old_variables:
                    sumScore += float(row[var])
                self.content[row['RID']]['cdrSum'] = sumScore
                self.content[row['RID']]['cdr'] = row['CDGLOBAL']

    def addColumns_tesla(self):
        self.columnNames.extend(['Tesla'])
        T15Table = self.readcsv('../../raw_tables/ADNI/MRIMETA.csv')
        T3Table = self.readcsv('../../raw_tables/ADNI/MRI3META.csv')
        for row in T15Table:
            if row['PHASE'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                self.content[row['RID']]['Tesla'] = row['FIELD_STRENGTH'][:-1]
        for row in T3Table:
            if row['PHASE'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                self.content[row['RID']]['Tesla'] = row['FIELD_STRENGTH'][:-1]+'.0'

    def addColumns_ADAS(self):
        variables = [ 'adas_q1','adas_q2','adas_q3','adas_q4','adas_q5','adas_q6','adas_q7','adas_q8','adas_q9','adas_q10','adas_q11','adas_q12','adas_q14', 'adas_total11', 'adas_totalmod']
        old_variables = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q14', 'TOTAL11', 'TOTALMOD']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ADASSCORES.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_TrailMaking(self):
        variables = ['trailA', 'trailB']
        old_variables = ['TMT_PtA_Complete', 'TMT_PtB_Complete']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ITEM.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['VISCODE'] == 'bl' and row['RID'] in self.content:
                if row['TMT_PtA_Complete'] and row['TMT_PtA_Complete'] != 'NULL' and 0 <= int(row['TMT_PtA_Complete']) <= 150:
                    self.content[row['RID']]['trailA'] = row['TMT_PtA_Complete']
                if row['TMT_PtB_Complete'] and row['TMT_PtB_Complete'] != 'NULL' and 0 <= int(row['TMT_PtB_Complete']) <= 300:
                    self.content[row['RID']]['trailB'] = row['TMT_PtB_Complete']

    def addColumns_logicalmemory(self):
        variables = ['lm_imm', 'lm_del', 'boston', 'animal', 'vege',
                     'digitB', 'digitBL', 'digitF', 'digitFL']
        old_variables = ['LIMMTOTAL', 'LDELTOTAL', 'BNTTOTAL', 'CATANIMSC', 'CATVEGESC']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/NEUROBAT.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase']=='ADNI1' and row['VISCODE'] in ['sc', 'bl'] and row['RID'] in self.content:
                if row['LIMMTOTAL'] and 0 <= int(row['LIMMTOTAL']) <= 25:
                    self.content[row['RID']]['lm_imm'] = row['LIMMTOTAL']
                if row['LDELTOTAL'] and 0 <= int(row['LDELTOTAL']) <= 25:
                    self.content[row['RID']]['lm_del'] = row['LDELTOTAL']
                if row['BNTTOTAL'] and 0 <= int(row['BNTTOTAL']) <= 30:
                    self.content[row['RID']]['boston'] = row['BNTTOTAL']
                if row['CATANIMSC'] and 0 <= int(row['CATANIMSC']) <= 77:
                    self.content[row['RID']]['animal'] = row['CATANIMSC']
                if row['CATVEGESC'] and 0 <= int(row['CATVEGESC']) <= 77:
                    self.content[row['RID']]['vege'] = row['CATVEGESC']
                if row['DSPANBAC'] and 0 <= int(row['DSPANBAC']) <= 12:
                    self.content[row['RID']]['digitB'] = row['DSPANBAC']
                if row['DSPANFOR'] and 0 <= int(row['DSPANFOR']) <= 12:
                    self.content[row['RID']]['digitF'] = row['DSPANFOR']
                if row['DSPANBLTH'] and 0 <= int(row['DSPANBLTH']) <= 8:
                    self.content[row['RID']]['digitBL'] = row['DSPANBLTH']
                if row['DSPANFLTH'] and 0 <= int(row['DSPANFLTH']) <= 8:
                    self.content[row['RID']]['digitFL'] = row['DSPANFLTH']

    def addColumns_Moca(self):
        variables = ['moca']
        old_variables = ['MOCA']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ADNIMERGE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['ORIGPROT'] == 'ADNI1' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_NPIQ(self):
        variables = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                     'npiq_ANX', 'npiq_ELAT', 'npiq_APA',  'npiq_DISN',
                     'npiq_IRR', 'npiq_MOT',  'npiq_NITE', 'npiq_APP']
        old_variables = ['NPIA', 'NPIB', 'NPIC', 'NPID',
                         'NPIE', 'NPIF', 'NPIG', 'NPIH',
                         'NPII', 'NPIJ', 'NPIK', 'NPIL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/NPIQ.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    if row[old_var] and row[old_var] == '0':
                        self.content[row['RID']][new_var] = '0'
                    elif row[old_var] and row[old_var] == '1':
                        self.content[row['RID']][new_var] = row[old_var+"SEV"]

    def addColumns_FAQ(self):
        variables = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                     'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
        old_variables = ['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG',
                         'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/FAQ.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_medhist(self):
        variables = ['his_CVHATT', 'his_PSYCDIS', 'his_Alcohol', 'his_SMOKYRS', 'his_PACKSPER']
        old_variables = ['MH4CARD', 'MHPSYCH', 'MH14ALCH', 'MH16BSMOK', 'MH16ASMOK']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/MEDHIST.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    if new_var == 'his_SMOKYRS':
                        if row[old_var] and row[old_var][0] != '-':
                            self.content[row['RID']][new_var] = row[old_var]
                    elif new_var == 'his_PACKSPER':
                        if row[old_var] and row[old_var] != '-4.0':
                            self.content[row['RID']][new_var] = float(row[old_var]) * 365
                    else:
                        self.content[row['RID']][new_var] = row[old_var]

    def addColumns_GDS(self):
        variables = ['gds']
        old_variables = ['GDTOTAL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/GDSCALE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                if row['GDTOTAL'] and 0 <= int(row['GDTOTAL']) <= 15:
                    self.content[row['RID']]['gds'] = row['GDTOTAL']

    def addColumns_FHQ(self):
        variables = ['his_NACCFAM']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/FHQ.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PHASE'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                if (row['FHQMOM']=='1' or row['FHQDAD']=='1' or row['FHQSIB']=='1'):
                    self.content[row['RID']]['his_NACCFAM'] = '1'
                else:
                    self.content[row['RID']]['his_NACCFAM'] = '0'

    def addColumns_modhach(self):
        variables = ['his_CBSTROKE', 'his_HYPERTEN']
        old_variables = ['HMSTROKE', 'HMHYPERT']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/MODHACH.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_dep(self):
        variables = ['his_DEPOTHR']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ADNI_DXSUM_PDXCONV.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI1' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                if row['DXNODEP'] == '1':
                    self.content[row['RID']]['his_DEPOTHR'] = '1'
                else:
                    self.content[row['RID']]['his_DEPOTHR'] = '0'

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_demograph()
        self.addColumns_apoe()
        self.addColumns_diagnosis()
        self.addColumns_mmse()
        self.addColumns_cdr()
        self.addColumns_tesla()
        self.addColumns_TrailMaking()
        self.addColumns_logicalmemory()
        # self.addColumns_ADAS()
        self.addColumns_NPIQ()
        self.addColumns_FAQ()
        self.addColumns_medhist()
        self.addColumns_FHQ()
        self.addColumns_modhach()
        self.addColumns_dep()
        self.addColumns_GDS()
        self.addColumns_Moca()
        with open(self.datasetName+'.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columnNames)
            writer.writeheader()
            for rid in self.content:
                writer.writerow(self.content[rid])

    def readcsv(self, csv_file):
        csvfile = open(csv_file, 'r')
        return csv.DictReader(csvfile)


# other useful comments
# PTID (patient id) is in the form of 123_S_5678
# the last 4 digits of PTID correspond to the RID (roster ID) which will be used for linking with other info
# DXCURREN 1 (NC) 2 (MCI) 3 (dementia and AD)
# reference https://adni.loni.usc.edu/wp-content/uploads/2012/08/slide_data_training_part2_reduced-size.pdf
# search unknown column name here: http://adni.loni.usc.edu/data-dictionary-search/

if __name__ == "__main__":
    obj = TableData()
    obj.writeTable()