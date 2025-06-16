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


def make_pred(dt: pd.DataFrame, source_info="kafka") -> pd.DataFrame:
    # Calculate score
    submission = pd.DataFrame({
        'score':  model.predict_proba(dt)[:, 1],
        'fraud_flag': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    logger.info(f'Prediction complete for data from {source_info}')

    # Return proba for positive class
    return submission