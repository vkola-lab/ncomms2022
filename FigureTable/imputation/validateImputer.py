import csv
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

train_data = "../../lookupcsv/CrossValid/cross0/train.csv"
test_data = "../../lookupcsv/CrossValid/cross0/test.csv"
target_col = 'education'
method = "KNN"

features = [
    "age", "gender", "education",
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

train_df = pd.read_csv(train_data)[features]
train_df = train_df.replace({'male': 0, 'female': 1})

# init the imputer
if method == 'KNN':
    imputer = KNNImputer(n_neighbors=10)
else:
    imputer = IterativeImputer(max_iter=1000)
imputer.fit(train_df)

# load test dataframe
test_df = pd.read_csv(test_data)[features]
test_df = test_df.replace({'male': 0, 'female': 1})
test_df = test_df.dropna(axis=0, how='any', thresh=None, subset=[target_col], inplace=False)

ground = copy.deepcopy(test_df[target_col])
xlim = [min(ground), max(ground)]

test_df[target_col] = np.nan
test_df = imputer.transform(test_df)
test_df = pd.DataFrame(test_df, columns=features)

pred = test_df[target_col]
ylim = [min(pred), max(pred)]

lim = [min(xlim[0], ylim[0]), max(xlim[0], xlim[1])]

print(ground, pred)

plt.scatter(ground, pred)
plt.ylim(lim)
plt.xlim(lim)
plt.savefig('scatter_{}.png'.format(target_col))
plt.close()




