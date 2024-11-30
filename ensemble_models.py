
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from scipy.stats import pearsonr
from datetime import datetime
from sklearn.base import TransformerMixin
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from desReg.des.DESRegression import DESRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer

import utils
import importlib
importlib.reload(utils)


class DenseTransformer(TransformerMixin):
    """
    Converte arrays esparsos para arrays densos.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, "toarray") else X


def avaliar_modelosCombinados_em_datasets(lista_datasets, versao_nome):
    # Definir os estimadores base para o DESRegression
    base_estimators = [
        KNeighborsRegressor(n_neighbors=3),
        KNeighborsRegressor(n_neighbors=5),
        DecisionTreeRegressor(max_depth=10),
        RandomForestRegressor(n_estimators=50),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1, max_iter=1000, tol=0.01),
        SVR(C=1.0, epsilon=0.1),
    ]

    # Definição dos modelos base
    models = [
        ('Linear Regression', LinearRegression(), None),
        ('Ridge Regression', Ridge(), {
         'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
        ('Lasso Regression', Lasso(max_iter=1000, tol=0.01), {
         'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}),
        ('ElasticNet Regression', ElasticNet(max_iter=1000), {
         'regressor__alpha': [0.1, 1.0, 10.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
        ('SVR', SVR(), {'regressor__C': [
         0.1, 1.0, 10.0], 'regressor__epsilon': [0.01, 0.1, 1.0]}),
        ('KNN', KNeighborsRegressor(), {
         'regressor__n_neighbors': [3, 5, 7, 9]}),
        ('Decision Tree', DecisionTreeRegressor(), {
         'regressor__max_depth': [None, 10, 20, 30]}),
        ('Random Forest', RandomForestRegressor(),
         {'regressor__n_estimators': [100, 200]}),
        ('XGBoost', XGBRegressor(), {'regressor__n_estimators': [
         100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.3]}),
        ('LightGBM', LGBMRegressor(), {'regressor__n_estimators': [
         100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.3]}),
        ('Stacking', StackingRegressor(estimators=[
            ('ridge', Ridge()),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('xgb', XGBRegressor(n_estimators=100))
        ], final_estimator=Ridge()), None),
        ('DESRegression', DESRegression(regressors_list=base_estimators,
         competence_region='knn', k=5, ensemble_type='DES'), None)
    ]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=5_000)),
        # Converte arrays esparsos para densos
        ('to_dense', DenseTransformer()),
        ('normalizer', Normalizer()),  # Normalizar os dados
        ('regressor', None)
    ])

    resultados_completos = []
    predicoes_por_modelo = {}

    for i, dados_filtrados in enumerate(lista_datasets):
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values
        issuekeys = dados_filtrados['issuekey'].values

        # Salvar ou carregar KFold específico para o dataset
        utils.salvar_kfold(dados_filtrados, dataset_name)
        kfold_indices = utils.carregar_kfold(dataset_name)

        # Resultados para armazenar as métricas
        results = {
            'Versao': [], 'Dataset': [], 'Model': [], 'MAE_Mean': [], 'MAE_Std': [],
            'R2_Mean': [], 'R2_Std': [], 'RMSE_Mean': [], 'RMSE_Std': [],
            'Pearson_Corr_Mean': [], 'Pearson_Corr_Std': [], 'Execution_DateTime': []
        }

        for name, model, param_grid in models:
            pipeline.set_params(regressor=model)
            print(f"\nModel: {name}")
            regressor = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='neg_mean_squared_error') if param_grid else pipeline

            list_maes_scores = []
            list_r2_scores = []
            list_rmse_scores = []
            list_pearson_scores = []
            all_y_test = []
            all_y_pred = []
            all_descriptions_test = []
            all_issuekeys_test = []

            for train_index, test_index in kfold_indices:
                X_train, X_test = descriptions[train_index], descriptions[test_index]
                y_train, y_test = effort_estimations[train_index], effort_estimations[test_index]
                issuekeys_test = issuekeys[test_index]

                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                try:
                    pearson_corr, _ = pearsonr(y_test, y_pred)
                except Exception as e:
                    print(f"Warning: {e}")
                    pearson_corr = float('nan')

                list_maes_scores.append(mae)
                list_r2_scores.append(r2)
                list_rmse_scores.append(rmse)
                list_pearson_scores.append(pearson_corr)

                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred)
                all_descriptions_test.extend(X_test)
                all_issuekeys_test.extend(issuekeys_test)

            predicoes_por_modelo[name] = {
                'issuekeys': all_issuekeys_test,
                'descriptions': all_descriptions_test,
                'y_test': all_y_test,
                'y_pred': all_y_pred
            }

            results['Versao'].append(versao_nome)
            results['Dataset'].append(dataset_name)
            results['Model'].append(name)
            results['MAE_Mean'].append(np.mean(list_maes_scores))
            results['MAE_Std'].append(np.std(list_maes_scores))
            results['R2_Mean'].append(np.mean(list_r2_scores))
            results['R2_Std'].append(np.std(list_r2_scores))
            results['RMSE_Mean'].append(np.mean(list_rmse_scores))
            results['RMSE_Std'].append(np.std(list_rmse_scores))
            results['Pearson_Corr_Mean'].append(np.mean(list_pearson_scores))
            results['Pearson_Corr_Std'].append(np.std(list_pearson_scores))
            results['Execution_DateTime'].append(datetime.now())

        resultados_completos.append(pd.DataFrame(results))

    resultados_finais = pd.concat(resultados_completos, ignore_index=True)

    return resultados_finais, predicoes_por_modelo
