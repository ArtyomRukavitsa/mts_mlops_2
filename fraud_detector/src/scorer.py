import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from catboost import Pool

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/my_model.cbm')

# Define optimal threshold
model_th = 0.5
logger.info('Pretrained model imported successfully...')
CAT_FEATURES = [
    'merch', 'cat_id', 'gender', 'one_city', 'us_state',
    'jobs', 'post_code', 'is_weekend', 'year', 'month',
    'day', 'hour', 'minute'
]

def make_pred(dt: pd.DataFrame, source_info="kafka") -> pd.DataFrame:
    # 1) Убираем target, если он есть
    if 'target' in dt.columns:
        dt = dt.drop(columns=['target'])
    
    # 2) Приводим датафрейм к тому же набору и порядку фич, что и при обучении
    expected_feats = model.feature_names_
    missing = set(expected_feats) - set(dt.columns)
    if missing:
        raise ValueError(f"В данных не хватает фич: {missing}")
    # Порядок очень важен :contentReference[oaicite:1]{index=1}
    dt = dt[expected_feats]
    
    # 3) Категории в str
    for c in CAT_FEATURES:
        if c in dt.columns:
            dt[c] = dt[c].astype(str)
    
    # 4) Собираем Pool с явным указанием категориальных
    pool = Pool(dt, cat_features=[c for c in CAT_FEATURES if c in dt.columns])
    
    # 5) Однократный вызов predict_proba
    proba = model.predict_proba(pool)[:, 1]
    flags = (proba > model_th).astype(int)
    
    submission = pd.DataFrame({
        'score': proba,
        'fraud_flag': flags
    })
    logger.info(f'Prediction complete for data from {source_info}')
    return submission