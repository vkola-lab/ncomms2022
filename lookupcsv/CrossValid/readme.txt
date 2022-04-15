this CrossValid folder will contain 5 fold cross validation split information

step1, run the combine.py to combine all subjects from all cohorts and output results in all.csv

step2, run the split.py to split the NACC into train, valid, test, and consider all other cohorts
as external testing, the data split results will be stored in folder cross0/ cross1/ etc

step3, run appendNonImage.py to fill up non-imaging information by table joining

