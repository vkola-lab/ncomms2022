import csv
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
from tabulate import tabulate

csv_file = "../../lookupcsv/dataset_table/NACC_ALL/NACC.csv"

vari_list = [
    "trailA", "trailB", "boston", "digitB", "digitBL", "digitF",
    "digitFL", "animal", "gds", "lm_imm", "lm_del", "mmse",
    "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX", "npiq_ELAT", "npiq_APA",
    "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP",
    "faq_BILLS", "faq_TAXES", "faq_SHOPPING", "faq_GAMES", "faq_STOVE",
    "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL",
    "his_NACCFAM", "his_CVHATT", "his_CVAFIB", "his_CVANGIO", "his_CVBYPASS", "his_CVPACE",
    "his_CVCHF", "his_CVOTHR", "his_CBSTROKE", "his_CBTIA", "his_SEIZURES", "his_TBI",
    "his_HYPERTEN", "his_HYPERCHO", "his_DIABETES", "his_B12DEF", "his_THYROID", "his_INCONTU", "his_INCONTF",
    "his_DEP2YRS", "his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL",
    "his_TOBAC100", "his_SMOKYRS", "his_PACKSPER", "his_ABUSOTHR"
]

label = 'ADD'

def simple_thresholding(target, label, sign=1):
    scores = []
    labels = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[target] and row[label]:
                scores.append(float(row[target]) * sign)
                labels.append(int(row[label]))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc_ = auc(fpr, tpr)
    ap_ = average_precision_score(labels, scores)
    if auc_ < 0.5:
        return simple_thresholding(target, label, sign=-1)
    else:
        return [target, "{:.3f}".format(auc_), "{:.3f}".format(ap_)]

content = []
for vari in vari_list:
    content.append(simple_thresholding(vari, label))
print(tabulate(content, headers=["variable", 'AUC', 'AP']))

with open("simple_ADD.csv", "w") as f:
    f.write(tabulate(content, headers=["variable", 'AUC', 'AP'], tablefmt="csv"))





