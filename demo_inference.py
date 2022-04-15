from model_wrappers import Multask_Wrapper
from utils import read_json

model = Multask_Wrapper(
    tasks=['ADD', 'COG'],
    device=1,
    main_config={"csv_dir": "lookupcsv/CrossValid/", "model_name": "CNN_baseline_new_cross0"},
    task_config=read_json('task_config.json'),
    seed=1000,
    loading_data=False
)
model.csv_dir = './demo/'
model.tb_log_dir = './demo/'
model.gen_score(['demo'])
