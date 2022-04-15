import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'ADNI3'
        self.imageDir = '/data/datasets/ADNI3/'
        self.imageFileNameList, self.RIDList = self.get_filenames_and_IDs(self.imageDir)
        self.columnNames = []
        self.content = defaultdict(dict) # dictionary of dictionary; {RID: {colname1: val1, colname2: val2, ...}}

    def get_filenames_and_IDs(self, path):
        fullpathList = glob(path + '*.nii')
        fileNameList = [fullpath.split('/')[-1] for fullpath in fullpathList]
        fileNameList = [a for a in fileNameList if a[11]=='6']
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
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                self.content[row['RID']]['age'] = int(row['USERDATE'][:4]) - int(row['PTDOBYY'])
                self.content[row['RID']]['gender'] = 'male' if row['PTGENDER'] == '1' else 'female'
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
            if row['RID'] in self.content:
                if (row['APGEN1'], row['APGEN2']) in [('3', '4'), ('4', '4')]:
                    self.content[row['RID']]['apoe'] = 1
                else:
                    self.content[row['RID']]['apoe'] = 0

    def addColumns_diagnosis(self):
        variables = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ADNI_DXSUM_PDXCONV.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                if row['DIAGNOSIS'] == '1':  # NL healthy control
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:  # Note that PD is not included here
                        self.content[row['RID']][var] = 0  # 0 means no
                    self.content[row['RID']]['NC'] = 1  # 1 means yes
                    self.content[row['RID']]['COG'] = 0
                elif row['DIAGNOSIS'] == '2':  # MCI patient
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:  # Note that PD is not included here, since all DXPARK=-4
                        self.content[row['RID']][var] = 0
                    self.content[row['RID']]['MCI'] = 1
                    self.content[row['RID']]['COG'] = 1
                elif row['DIAGNOSIS'] == '3':  # Dementia patient
                    for var in ['NC', 'MCI']:
                        self.content[row['RID']][var] = 0
                    self.content[row['RID']]['DE'] = 1
                    self.content[row['RID']]['COG'] = 2
                    self.content[row['RID']]['ADD'] = 1
                    if row['DXDDUE'] == '1': # dementia due to AD
                        self.content[row['RID']]['AD'] = 1
                    else: # value 2 means dementia due to ether etiologies
                        pass # turns out all 'DXDDUE'=='1'for dementia cases

    def addColumns_mmse(self):
        variables = ['mmse', 'mmse_orient', 'mmse_workMem', 'mmse_concen', 'mmse_memRec', 'mmse_lang', 'mmse_visuSpa']
        old_variables = ['mmse','Orientation','Working Memory','Concentration','Memory Recall','Language','Visuospatial']
        variables = ['mmse']
        old_variables = ['mmse']
        self.columnNames.extend(variables)
        targetTable = '../../derived_tables/ADNI/ADNI_MMSE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_cdr(self):
        variables = ['cdr', 'cdrSum']
        old_variables = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/CDR.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                sumScore = 0
                for var in old_variables:
                    sumScore += float(row[var])
                self.content[row['RID']]['cdrSum'] = sumScore
                self.content[row['RID']]['cdr'] = row['CDGLOBAL']

    def addColumns_moca(self):
        variables = ['moca','moca_execu','moca_visuo','moca_name','moca_atten','moca_senrep','moca_verba','moca_abstr','moca_delrec','moca_orient']
        old_variables = ['moca','moca_execu','moca_visuo','moca_name','moca_atten','moca_senrep','moca_verba','moca_abstr','moca_delrec','moca_orient']
        variables = ['moca']
        old_variables = ['moca']
        self.columnNames.extend(variables)
        targetTable = '../../derived_tables/ADNI/ADNIGO_MOCA.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_logicalmemory(self):
        variables = ['lm_imm', 'lm_del', 'boston', 'animal', 'vege',
                     'digitB', 'digitBL', 'digitF', 'digitFL']
        old_variables = ['LIMMTOTAL', 'LDELTOTAL', 'BNTTOTAL', 'CATANIMSC', 'CATVEGESC']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/NEUROBAT.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] in ['sc', 'bl'] and row['RID'] in self.content:
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

    def addColumns_NPIQ(self):
        variables = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                     'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN',
                     'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP']
        old_variables = ['NPIA', 'NPIB', 'NPIC', 'NPID',
                         'NPIE', 'NPIF', 'NPIG', 'NPIH',
                         'NPII', 'NPIJ', 'NPIK', 'NPIL']
        sev_variables = ['NPIA10B', 'NPIB8B', 'NPIC9B', 'NPID9B',
                         'NPIE8A', 'NPIF8B', 'NPIG9B', 'NPIH8B',
                         'NPII8B', 'NPIJ8B', 'NPIK9B', 'NPIL9B']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/NPI.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE2'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var, sev_var in zip(old_variables, variables, sev_variables):
                    if row[old_var] and row[old_var] == '0':
                        self.content[row['RID']][new_var] = '0'
                    elif row[old_var] and row[old_var] == '1':
                        self.content[row['RID']][new_var] = row[sev_var]

    def addColumns_FAQ(self):
        variables = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                     'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
        old_variables = ['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG',
                         'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/FAQ.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE2'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_GDS(self):
        variables = ['gds']
        old_variables = ['GDTOTAL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/GDSCALE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE2'] == 'sc' and row['RID'] in self.content:
                if row['GDTOTAL'] and 0 <= int(row['GDTOTAL']) <= 15:
                    self.content[row['RID']]['gds'] = row['GDTOTAL']

    def addColumns_medhist(self):
        variables = ['his_CVHATT', 'his_PSYCDIS', 'his_Alcohol', 'his_SMOKYRS', 'his_PACKSPER']
        old_variables = ['MH4CARD', 'MHPSYCH', 'MH14ALCH', 'MH16BSMOK', 'MH16ASMOK']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/MEDHIST.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    if new_var == 'his_SMOKYRS':
                        if row[old_var][0] != '-':
                            self.content[row['RID']][new_var] = row[old_var]
                    elif new_var == 'his_PACKSPER':
                        if row[old_var] != '-4.0':
                            self.content[row['RID']][new_var] = float(row[old_var]) * 365
                    else:
                        self.content[row['RID']][new_var] = row[old_var]

    def addColumns_FHQ(self):
        variables = ['his_NACCFAM']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/FAMXHPAR.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                if row['MOTHDEM'] == '1' or row['FATHDEM']:
                    self.content[row['RID']]['his_NACCFAM'] = '1'
                else:
                    self.content[row['RID']]['his_NACCFAM'] = '0'
        targetTable = '../../raw_tables/ADNI/FAMXHSIB.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                if self.content[row['RID']]['his_NACCFAM'] == '1':
                    continue
                if row['SIBDEMENT'] == '1':
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
            if row['Phase'] == 'ADNI3' and row['VISCODE2'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_dep(self):
        variables = ['his_DEPOTHR']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/BLCHANGE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNI3' and row['VISCODE2'] in ['bl', 'sc'] and row['RID'] in self.content:
                self.content[row['RID']]['his_DEPOTHR'] = row['BCDEPRES']

    def addColumns_tesla(self):
        self.columnNames.append('Tesla')
        for row in self.content:
            self.content[row]['Tesla'] = '3.0'

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_demograph()
        self.addColumns_apoe()
        self.addColumns_diagnosis()
        self.addColumns_mmse()
        self.addColumns_cdr()
        self.addColumns_moca()
        self.addColumns_logicalmemory()
        self.addColumns_NPIQ()
        self.addColumns_FAQ()
        self.addColumns_GDS()
        self.addColumns_medhist()
        self.addColumns_FHQ()
        self.addColumns_modhach()
        self.addColumns_dep()
        self.addColumns_tesla()
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