import os
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from joblib import load
import scipy.sparse


# ________________________________________Tests model____________________________________

path = "D:\\OneDrive\\DATA\\FORMATIONS\\MLOps\\projet mercari\\Projet_mlops_Mercarichallenge\\API\\"


# Test du chargement des fichiers de données
def test_data_loading():
    X_train = scipy.sparse.load_npz(path + "train_final.npz")
    y_train = np.load(path+'y_train.npy')

    X_cv = scipy.sparse.load_npz(path + r"\cv_final.npz")
    y_cv = np.load(path+'y_cv.npy')

    assert X_train.shape[0] == y_train.shape[0], "Les dimensions de X_train et y_train ne correspondent pas"
    assert X_cv.shape[0] == y_cv.shape[0], "Les dimensions de X_cv et y_cv ne correspondent pas"

# Test de présence du model (fichier joblib)
def test_model_saving():
    assert os.path.isfile('model_svr.joblib'), "Le modèle n'a pas été sauvegardé correctement"

# Test de la prediction du model
def test_model_prediction():
    model = load('model_svr.joblib')
    X_cv = scipy.sparse.load_npz(path + r"\cv_final.npz")
    preds = model.predict(X_cv)
    assert preds.shape[0] == X_cv.shape[0], "Les dimensions des prédictions et de X_cv ne correspondent pas"

# Evaluation du model
def test_model_evaluation():
    model = load('model_svr.joblib')
    X_cv = scipy.sparse.load_npz(path + r"\cv_final.npz")
    y_cv = np.load(path+'y_cv.npy')

    lgb_preds_cv = model.predict(X_cv)
    rmse_score = sqrt(mse(y_cv, lgb_preds_cv))

    assert rmse_score > 0, "Le score RMSE devrait être supérieur à 0"
    assert rmse_score < 1, "Le score RMSE devrait être inférieur à 1"