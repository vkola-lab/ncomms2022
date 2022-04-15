import csv
from glob import glob
from collections import defaultdict

class TableData:
    def __init__(self):
        self.datasetName = 'ADNIGO'
        self.imageDir = '/data/datasets/ADNIGO/'
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
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
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

    def addColumns_diagnosis(self):
        variables = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']
        self.columnNames.extend(variables)
        # according to table '../../raw_tables/ADNIGO/ADNI_GO.csv' all cases are MCI
        for rid in self.content:
            for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                self.content[rid][var] = 0
            self.content[rid]['MCI'] = 1
            self.content[rid]['COG'] = 1
        # parkinson label not sure here, leave as blank

    def addColumns_mmse(self):
        variables = ['mmse', 'mmse_orient', 'mmse_workMem', 'mmse_concen', 'mmse_memRec', 'mmse_lang', 'mmse_visuSpa']
        old_variables = ['mmse','Orientation','Working Memory','Concentration','Memory Recall','Language','Visuospatial']
        variables = ['mmse']
        old_variables = ['mmse']
        self.columnNames.extend(variables)
        targetTable = '../../derived_tables/ADNI/ADNI_MMSE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]
                    
    def addColumns_nb(self):
        variables = ['nb_CLOCKSCOR', 'nb_COPYSCOR', 'nb_AVTOTB', 'nb_CATANIMSC', 'nb_TRAASCOR', 'nb_TRABSCOR', 'nb_BNTTOTAL', 'nb_AVDEL30MIN', 'nb_AVDELTOT','nb_ANARTERR']
        old_variables = ['CLOCKSCOR', 'COPYSCOR','AVTOTB','CATANIMSC', 'TRAASCOR', 'TRABSCOR', 'BNTTOTAL', 'AVDEL30MIN', 'AVDELTOT', 'ANARTERR']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/NEUROBAT.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]
                    
    def addColumns_cdr(self):
        variables = ['cdr', 'cdrSum']
        old_variables = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/CDR.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                sumScore = 0
                for var in old_variables:
                    sumScore += float(row[var])
                self.content[row['RID']]['cdrSum'] = sumScore
                self.content[row['RID']]['cdr'] = row['CDGLOBAL']

    def addColumns_logicalmemory(self):
        variables = ['lm_imm', 'lm_del', 'boston', 'animal', 'vege',
                     'digitB', 'digitBL', 'digitF', 'digitFL']
        old_variables = ['LIMMTOTAL', 'LDELTOTAL', 'BNTTOTAL', 'CATANIMSC', 'CATVEGESC']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/NEUROBAT.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase']=='ADNIGO' and row['VISCODE'] in ['sc', 'bl'] and row['RID'] in self.content:
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

    def addColumns_moca(self):
        variables = ['moca','moca_execu','moca_visuo','moca_name','moca_atten','moca_senrep','moca_verba','moca_abstr','moca_delrec','moca_orient']
        old_variables = ['moca','moca_execu','moca_visuo','moca_name','moca_atten','moca_senrep','moca_verba','moca_abstr','moca_delrec','moca_orient']
        variables = ['moca']
        old_variables = ['moca']
        self.columnNames.extend(variables)
        targetTable = '../../derived_tables/ADNI/ADNIGO_MOCA.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]
                    
    def addColumns_adas(self):
        variables = ['adas_Q1score', 'adas_Q2score', 'adas_Q3score', 'adas_Q4score', 'adas_Q5score', 'adas_Q6score', 'adas_Q7score', 'adas_Q8score', 'adas_Q9score', 'adas_Q10score', 'adas_Q11score', 'adas_Q12score', 'adas_Q13score', 'adas_TOTAL13', 'adas_TOTAL_Score']
        old_variables = ['Q1SCORE', 'Q2SCORE', 'Q3SCORE', 'Q4SCORE', 'Q5SCORE', 'Q6SCORE', 'Q7SCORE', 'Q8SCORE', 'Q9SCORE', 'Q10SCORE', 'Q11SCORE', 'Q12SCORE', 'Q13SCORE', 'TOTAL13', 'TOTSCORE']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/ADAS_ADNIGO23.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]
    
    def addColumns_tesla(self):
        self.columnNames.extend(['Tesla'])
        T15Table = self.readcsv('../../raw_tables/ADNI/MRIMETA.csv')
        T3Table = self.readcsv('../../raw_tables/ADNI/MRI3META.csv')
        for row in T15Table:
            if row['PHASE'] == 'ADNIGO' and row['VISCODE'] == 'scmri' and row['RID'] in self.content:
                self.content[row['RID']]['Tesla'] = row['FIELD_STRENGTH'][:-1]
        for row in T3Table:
            if row['PHASE'] == 'ADNIGO' and row['VISCODE'] == 'scmri' and row['RID'] in self.content:
                self.content[row['RID']]['Tesla'] = row['FIELD_STRENGTH'][:-1]+'.0'

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

    def addColumns_GDS(self):
        variables = ['gds']
        old_variables = ['GDTOTAL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/GDSCALE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
                if row['GDTOTAL'] and 0 <= int(row['GDTOTAL']) <= 15:
                    self.content[row['RID']]['gds'] = row['GDTOTAL']

    def addColumns_FAQ(self):
        variables = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                     'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
        old_variables = ['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG',
                         'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/FAQ.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE2'] == 'bl' and row['RID'] in self.content:
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
            if row['Phase'] == 'ADNIGO' and row['VISCODE'] == 'bl' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    if row[old_var] and row[old_var] == '0':
                        self.content[row['RID']][new_var] = '0'
                    elif row[old_var] and row[old_var] == '1':
                        self.content[row['RID']][new_var] = row[old_var+"SEV"]

    def addColumns_medhist(self):
        variables = ['his_CVHATT', 'his_PSYCDIS', 'his_Alcohol', 'his_SMOKYRS', 'his_PACKSPER']
        old_variables = ['MH4CARD', 'MHPSYCH', 'MH14ALCH', 'MH16BSMOK', 'MH16ASMOK']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/MEDHIST.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE2'] == 'sc' and row['RID'] in self.content:
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
        targetTable = '../../raw_tables/ADNI/FHQ.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['PHASE'] == 'ADNIGO' and row['VISCODE'] == 'sc' and row['RID'] in self.content:
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
            if row['Phase'] == 'ADNIGO' and row['VISCODE2'] == 'sc' and row['RID'] in self.content:
                for old_var, new_var in zip(old_variables, variables):
                    self.content[row['RID']][new_var] = row[old_var]

    def addColumns_dep(self):
        variables = ['his_DEPOTHR']
        self.columnNames.extend(variables)
        targetTable = '../../raw_tables/ADNI/BLCHANGE.csv'
        targetTable = self.readcsv(targetTable)
        for row in targetTable:
            if row['Phase'] == 'ADNIGO' and row['VISCODE2'] in ['bl', 'sc'] and row['RID'] in self.content:
                self.content[row['RID']]['his_DEPOTHR'] = row['BCDEPRES']

    def writeTable(self):
        self.addColumns_path_filename()
        self.addColumns_demograph()
        self.addColumns_apoe()
        self.addColumns_diagnosis()
        self.addColumns_mmse()
        self.addColumns_logicalmemory()
        # self.addColumns_nb()
        self.addColumns_cdr()
        self.addColumns_moca()
        # self.addColumns_adas()
        self.addColumns_NPIQ()
        self.addColumns_FAQ()
        self.addColumns_GDS()
        self.addColumns_medhist()
        self.addColumns_FHQ()
        self.addColumns_modhach()
        self.addColumns_dep()
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



