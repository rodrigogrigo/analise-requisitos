from lightgbm import LGBMRegressor

import utils
import numpy as np
import sys
import pandas as pd
import preprocessing

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from desReg.des.DESRegression import DESRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
from datetime import datetime


def get_models() -> list:

    base_estimators = [
        KNeighborsRegressor(n_neighbors=3),
        DecisionTreeRegressor(max_depth=10),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1, max_iter=1000, tol=0.01)
    ]

    return [
        ('Linear Regression', LinearRegression()),
        ('Ridge Regression', Ridge(alpha=1.0)),
        ('Lasso Regression', Lasso(alpha=0.1, max_iter=1000, tol=0.01)),
        ('ElasticNet Regression', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)),
        ('SVR', SVR(C=1.0, epsilon=0.1)),
        ('KNN', KNeighborsRegressor(n_neighbors=5)),
        ('Decision Tree', DecisionTreeRegressor(max_depth=10)),
        ('Random Forest', RandomForestRegressor(n_estimators=100)),
        ('XGBoost', XGBRegressor(n_estimators=100)),
        ('LightGBM', LGBMRegressor(n_estimators=100)),
        ('Stacking', StackingRegressor(estimators=[
            ('ridge', Ridge()),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('xgb', XGBRegressor(n_estimators=100))
        ], final_estimator=Ridge())),
        ('DESRegression', DESRegression(regressors_list=base_estimators,
                                        competence_region='knn', k=5, ensemble_type='DES'))
    ]


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, "toarray") else X


def create_pipeline(model):
    """
    Cria um pipeline com TF-IDF, transformação para denso, normalização e regressor.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=5_000)),
        ('to_dense', DenseTransformer()),
        ('regressor', model)
    ])


def evaluate_model(pipeline, x_train, y_train, x_test, y_test):
    """
    Treina o pipeline com X_train e y_train, avalia a performance no conjunto de validação,
    depois re-treina com o conjunto combinado (treinamento + validação) e avalia no conjunto de teste.

    O scaler é ajustado (fit) com os targets para poder usar inverse_transform.

    Retorna:
      - Métricas no conjunto de validação: mae_val, r2_val, rmse_val, pearson_val
      - Métricas no conjunto de teste: mae_test, r2_test, rmse_test, pearson_test
    """

    # --- Avaliação no conjunto de validação ---

    pipeline.fit(x_train, y_train)

    y_test_pred = pipeline.predict(x_test)
    y_test_pred = np.array(y_test_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred)
    pearson_test, _ = pearsonr(y_test, y_test_pred)

    return mae_test, r2_test, rmse_test, pearson_test


def avaliar_modelos_intra_datasets(lista_datasets: list, versao_nome: str,
                                   nome_arquivo_csv: str) -> pd.DataFrame:
    """
        Avalia os modelos usando KFold.
        Para cada dataset e para cada modelo, divide os dados (usando os folds previamente salvos),
        separa internamente cada fold em treinamento e validação e depois re-treina com a junção
        dos dados de treinamento e validação para testar no conjunto de teste do fold.

        Os resultados (médias e desvios) de cada modelo são exportados uma única vez, por dataset.
    """

    resultados_completos = []

    models = get_models()

    for i, dados_filtrados in enumerate(lista_datasets):

        dataset_name = dados_filtrados['dataset_name'].iloc[0]

        print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')

        descriptions = np.array(dados_filtrados['treated_description_ml'].values)
        effort_estimations = np.array(dados_filtrados['storypoint'].values)

        utils.salvar_kfold(dados_filtrados, dataset_name)

        kfold_indices = utils.carregar_kfold(dataset_name)

        # Para cada modelo, acumula os resultados de cada fold

        for (model_name, model) in models:

            print(f"\n\tIniciando o modelo: {model_name}\n")

            mae_test_list = []
            r2_test_list = []
            rmse_test_list = []
            pearson_test_list = []

            # Processa cada fold

            with tqdm(total=len(kfold_indices), file=sys.stdout, colour='blue',
                      desc=f"\t\tProcessando Folds para {model_name}") as fold_bar:

                for train_index, test_index in kfold_indices:

                    X_train = descriptions[train_index]
                    y_train = effort_estimations[train_index]
                    X_test = descriptions[test_index]
                    y_test = effort_estimations[test_index]

                    # Divisão interna do fold em treinamento e validação
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )

                    pipeline = create_pipeline(model)

                    (mae_test, r2_test, rmse_test, pearson_test) = evaluate_model(
                        pipeline, X_train, y_train, X_test, y_test)

                    mae_test_list.append(mae_test)
                    r2_test_list.append(r2_test)
                    rmse_test_list.append(rmse_test)
                    pearson_test_list.append(pearson_test)

                    fold_bar.update(1)

            # Agregação dos resultados dos folds (média e desvio padrão)
            result = {
                'Versao': versao_nome,
                'Dataset': dataset_name,
                'Model': model_name,
                'MAE_Test_Mean': np.mean(mae_test_list),
                'MAE_Test_Std': np.std(mae_test_list),
                'R2_Test_Mean': np.mean(r2_test_list),
                'R2_Test_Std': np.std(r2_test_list),
                'RMSE_Test_Mean': np.mean(rmse_test_list),
                'RMSE_Test_Std': np.std(rmse_test_list),
                'Pearson_Test_Mean': np.mean(pearson_test_list),
                'Pearson_Test_Std': np.std(pearson_test_list),
                'Execution_DateTime': datetime.now()
            }

            # Exporta os resultados para este modelo (por este dataset) uma única vez
            resultados_df_model = pd.DataFrame([result])
            resultados_df_model.drop_duplicates(inplace=True)
            preprocessing.exportar_resultados_para_csv(resultados_df_model, nome_arquivo_csv)
            resultados_completos.append(result)

    print("\n\nProcessamento finalizado!")

    return pd.DataFrame(resultados_completos)


def avaliar_modelos_inter_datasets(lista_datasets: list, versao_nome: str,
                                   nome_arquivo_csv: str) -> pd.DataFrame:

    resultados_completos = []

    n_datasets = len(lista_datasets)
    models = get_models()

    # Itera sobre os modelos

    for (model_name, model) in models:

        print(f"\n=== Processando modelo: {model_name} ===")

        temp_results = []  # resultados deste modelo (across all cycles)

        # Processa cada ciclo (cada índice i na lista de datasets)
        for test_id in range(n_datasets):

            # Seleção dos datasets de forma cíclica

            dataset_teste = lista_datasets[test_id]
            dataset_validacao = lista_datasets[(test_id + 1) % n_datasets]

            # Extração dos nomes para log
            nome_test = dataset_teste['dataset_name'].iloc[0]
            nome_val = dataset_validacao['dataset_name'].iloc[0]

            X_train = []
            y_train = []

            # Cria a string que representa os datasets de treinamento
            dataset_treino_comb = ''

            for train_id in range(n_datasets):

                ds_dataset = lista_datasets[train_id]
                ds_name = ds_dataset["dataset_name"].iloc[0]

                if ds_name != nome_test and ds_name != nome_val:

                    dataset_treino_comb += f"{ds_name}_"

                    if len(X_train) == 0:
                        X_train = lista_datasets[train_id]["treated_description_ml"].values
                        y_train = lista_datasets[train_id]["storypoint"].values
                    else:
                        X_train = np.concatenate([
                            X_train,
                            lista_datasets[train_id]["treated_description_ml"].values
                        ])
                        y_train = np.concatenate([
                            y_train,
                            lista_datasets[train_id]["storypoint"].values
                        ])

            dataset_treino_comb = dataset_treino_comb[:-1]

            # Verifica se essa combinação já foi processada
            if utils.combinacao_ja_avaliada_inter(nome_arquivo_csv, versao_nome, dataset_treino_comb,
                                                  nome_val, nome_test, model_name):
                print(f"\t[IGNORADO] Já existe resultado para esta configuração.")
                continue

            print(f"\n\tCiclo {test_id + 1}/{n_datasets} -- Treinamento: {dataset_treino_comb}; "
                  f"Validação: {nome_val}; Teste: {nome_test}")


            # Dados para validação
            _ = np.array(dataset_validacao['treated_description_ml'].values)
            _ = np.array(dataset_validacao['storypoint'].values)

            # Dados para teste
            X_test = np.array(dataset_teste['treated_description_ml'].values)
            y_test = np.array(dataset_teste['storypoint'].values)

            pipeline = create_pipeline(model)

            mae_test, r2_test, rmse_test, pearson_test = evaluate_model(
                pipeline, X_train, y_train, X_test, y_test)

            result = {
                'Versao': versao_nome,
                'Treinamento': f'{dataset_treino_comb}',
                'Validacao': nome_val,
                'Teste': nome_test,
                'Model': model_name,
                'MAE_Test': mae_test,
                'R2_Test': r2_test,
                'RMSE_Test': rmse_test,
                'Pearson_Test': pearson_test,
                'Execution_DateTime': datetime.now()
            }

            temp_results.append(result)

        # Ao terminar todos os ciclos para o modelo, exporta os resultados deste modelo

        resultados_df_model = pd.DataFrame(temp_results)

        resultados_df_model.drop_duplicates(inplace=True)
        preprocessing.exportar_resultados_para_csv(resultados_df_model, nome_arquivo_csv)

        print(f"\nResultados exportados para o modelo: {model_name}")
        resultados_completos.extend(temp_results)

    print("\n\nProcessamento finalizado!")

    return pd.DataFrame(resultados_completos)
