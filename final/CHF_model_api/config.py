
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "../logs")
MODELS_DIR = os.path.join(BASE_DIR, "../saved_models")
HPTB_DIR = os.path.join(BASE_DIR, "../hparams_tuning_tb")
CSV_DIR = os.path.join(BASE_DIR, "../csv_files")
SQLDB_DIR = os.path.join(BASE_DIR, "../optuna_sql_databases")
PDF_DIR = os.path.join(BASE_DIR, "../pdf_files")
VISU_DIR = os.path.join(BASE_DIR, "../visuals")





REMOVE_NEG_DHIN = True

SEED_NP = 4
SEED_RAND = 5

TEST_DATA_PROPORTION = 0.2