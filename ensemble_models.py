from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from datetime import datetime
from sklearn.base import TransformerMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from desReg.des.DESRegression import DESRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, StandardScaler
import optuna  # Adicionando Optuna para otimização de hiperparâmetros

import utils
import preprocessing

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
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=5000)),
        ('to_dense', DenseTransformer()),
        ('normalizer', Normalizer()),
        ('regressor', model)
    ])

def normalize_data(scaler, datasets):
    """
    Normaliza os story points nos datasets fornecidos,
    sempre ajustando (fit) o scaler para cada dataset.
    """
    for dataset in datasets:
        dataset['storypoint_scaled'] = scaler.fit_transform(dataset[['storypoint']])

def evaluate_model(pipeline, X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """
    Treina o pipeline com X_train e y_train, avalia a performance no conjunto de validação,
    depois re-treina com o conjunto combinado (treinamento + validação) e avalia no conjunto de teste.
    
    O scaler é ajustado (fit) com os targets para poder usar inverse_transform.
    
    Retorna:
      - Métricas no conjunto de validação: mae_val, r2_val, rmse_val, pearson_val
      - Métricas no conjunto de teste: mae_test, r2_test, rmse_test, pearson_test
    """
    # --- Avaliação no conjunto de validação ---
    pipeline.fit(X_train, y_train)
    scaler.fit(y_train.reshape(-1, 1))
    
    y_val_pred = pipeline.predict(X_val)
    y_val_pred = np.array(y_val_pred).reshape(-1, 1)
    
    y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_val_pred_original = scaler.inverse_transform(y_val_pred).flatten()
    
    mae_val = mean_absolute_error(y_val_original, y_val_pred_original)
    r2_val = r2_score(y_val_original, y_val_pred_original)
    rmse_val = mean_squared_error(y_val_original, y_val_pred_original, squared=False)
    pearson_val, _ = pearsonr(y_val_original, y_val_pred_original)
    
    # --- Avaliação no conjunto de teste (re-treinando com treinamento + validação) ---
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    pipeline.fit(X_train_val, y_train_val)
    scaler.fit(y_train_val.reshape(-1, 1))
    
    y_test_pred = pipeline.predict(X_test)
    y_test_pred = np.array(y_test_pred).reshape(-1, 1)
    
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_original = scaler.inverse_transform(y_test_pred).flatten()
    
    mae_test = mean_absolute_error(y_test_original, y_test_pred_original)
    r2_test = r2_score(y_test_original, y_test_pred_original)
    rmse_test = mean_squared_error(y_test_original, y_test_pred_original, squared=False)
    pearson_test, _ = pearsonr(y_test_original, y_test_pred_original)
    
    return mae_val, r2_val, rmse_val, pearson_val, mae_test, r2_test, rmse_test, pearson_test

#def evaluate_model(pipeline, X_train, y_train, X_test, y_test, scaler):
#    # Ajustar o pipeline aos dados de treinamento
#    pipeline.fit(X_train, y_train)
#
#    # Treinar o scaler antes de usar inverse_transform
#    scaler.fit(y_train.reshape(-1, 1))  
#
#    # Fazer a predição
#    y_pred = pipeline.predict(X_test)
#    y_pred = np.array(y_pred).reshape(-1, 1)
#
#    # Desnormalizar os resultados
#    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
#    y_pred_original = scaler.inverse_transform(y_pred).flatten()
#
#    # Calcular métricas
#    mae = mean_absolute_error(y_test_original, y_pred_original)
#    r2 = r2_score(y_test_original, y_pred_original)
#    rmse = mean_squared_error(y_test_original, y_pred_original, squared=False)
#    pearson_corr, _ = pearsonr(y_test_original, y_pred_original)
#
#    return mae, r2, rmse, pearson_corr

def objective(trial, X_train, y_train, X_val, y_val, model_name):
    """
    Função objetivo para otimização com Optuna.
    Define os hiperparâmetros a serem otimizados para cada modelo e retorna a métrica de validação (MAE).
    """
    if model_name == 'Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 10, 30, step=5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    elif model_name == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    else:
        raise ValueError(f"Modelo {model_name} não suportado para otimização com Optuna.")

    pipeline = create_pipeline(model)
    # Aqui, utilizamos a parte de validação da função auxiliar; como não há conjunto de teste nesta etapa,
    # podemos passar X_val e y_val também como "teste" – o importante é retornar o erro de validação.
    mae_val, _, _, _, _, _, _, _ = evaluate_model(pipeline, X_train, y_train, X_val, y_val, X_val, y_val, StandardScaler())
    return mae_val

#def objective(trial, X_train, y_train, X_test, y_test, model_name):
#   """
#    Função objetivo para otimização com Optuna.
#    Define os hiperparâmetros a serem otimizados para cada modelo.
#    """
#    if model_name == 'Random Forest':
#        n_estimators = trial.suggest_int('n_estimators', 50, 200)
#        max_depth = trial.suggest_int('max_depth', 10, 30, step=5)
#        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
#    elif model_name == 'XGBoost':
#        n_estimators = trial.suggest_int('n_estimators', 50, 200)
#        max_depth = trial.suggest_int('max_depth', 3, 10)
#        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
#        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
#    elif model_name == 'LightGBM':
#        n_estimators = trial.suggest_int('n_estimators', 50, 200)
#        max_depth = trial.suggest_int('max_depth', 3, 10)
#        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
#        model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
#    else:
#        raise ValueError(f"Modelo {model_name} não suportado para otimização com Optuna.")
#
#    pipeline = create_pipeline(model)
#    mae = evaluate_model(pipeline, X_train, y_train, X_test, y_test, StandardScaler())[0]
#    return mae

def get_models():
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
        ('Random Forest', 'Random Forest'),
        ('XGBoost', 'XGBoost'),
        ('LightGBM', 'LightGBM'),
        ('Stacking', StackingRegressor(estimators=[
            ('ridge', Ridge()),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('xgb', XGBRegressor(n_estimators=100))
        ], final_estimator=Ridge())),
        ('DESRegression', DESRegression(regressors_list=base_estimators, competence_region='knn', k=5, ensemble_type='DES'))
    ]

def avaliar_modelos_em_datasets_ciclico(lista_datasets, versao_nome, nome_arquivo_csv):
    """
    Avaliação cíclica:
      - Treinamento: datasets i e i+1 (concatenados)
      - Validação: dataset i+2
      - Teste: dataset i+3

    Para cada modelo, todos os ciclos são processados e, ao final, os resultados daquele modelo são exportados
    (sem repetição) e adicionados ao conjunto global de resultados.
    """
    resultados_completos = []
    n = len(lista_datasets)
    models = get_models()
    
    # Itera sobre os modelos
    for (model_name, model) in models:
        print(f"\n=== Processando modelo: {model_name} ===")
        temp_results = []  # resultados deste modelo (across all cycles)
        
        # Processa cada ciclo (cada índice i na lista de datasets)
        for i in range(n):
            # Seleção dos datasets de forma cíclica
            dataset_train1 = lista_datasets[i]
            dataset_train2 = lista_datasets[(i + 1) % n]
            dataset_validacao = lista_datasets[(i + 2) % n]
            dataset_teste = lista_datasets[(i + 3) % n]
            
            # Extração dos nomes para log
            nome_train1 = dataset_train1['dataset_name'].iloc[0]
            nome_train2 = dataset_train2['dataset_name'].iloc[0]
            nome_validacao = dataset_validacao['dataset_name'].iloc[0]
            nome_teste = dataset_teste['dataset_name'].iloc[0]
            print(f"\nCiclo {i+1}/{n} -- Treinamento: {nome_train1} e {nome_train2}; Validação: {nome_validacao}; Teste: {nome_teste}")
            
            # Cria uma nova instância de scaler para este ciclo (para evitar interferência entre ciclos)
            scaler = StandardScaler()
            normalize_data(scaler, [dataset_train1, dataset_train2, dataset_validacao, dataset_teste])
            
            # Concatena os dados dos dois conjuntos de treinamento
            X_train = np.concatenate([
                np.array(dataset_train1['treated_description'].values),
                np.array(dataset_train2['treated_description'].values)
            ])
            y_train = np.concatenate([
                np.array(dataset_train1['storypoint_scaled'].values),
                np.array(dataset_train2['storypoint_scaled'].values)
            ])
            
            # Dados para validação
            X_val = np.array(dataset_validacao['treated_description'].values)
            y_val = np.array(dataset_validacao['storypoint_scaled'].values)
            
            # Dados para teste
            X_test = np.array(dataset_teste['treated_description'].values)
            y_test = np.array(dataset_teste['storypoint_scaled'].values)
            
            # Se o modelo usa Optuna, otimize os hiperparâmetros usando os dados de treinamento e validação
            best_params = 'N/A'
            if isinstance(model, str):
                print(f"Otimizando e avaliando {model_name} neste ciclo...")
                # Aqui, usamos os dados de treinamento (X_train, y_train) e validação (X_val, y_val)
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, model),
                               n_trials=5, n_jobs=-1)
                best_params = study.best_params
                print(f"Melhores hiperparâmetros para {model_name} neste ciclo: {best_params}")
                if model == 'Random Forest':
                    model_instance = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
                elif model == 'XGBoost':
                    model_instance = XGBRegressor(**best_params, random_state=42)
                elif model == 'LightGBM':
                    model_instance = LGBMRegressor(**best_params, random_state=42)
                else:
                    model_instance = None
            else:
                model_instance = model
            
            pipeline = create_pipeline(model_instance)
            (mae_val, r2_val, rmse_val, pearson_val,
            mae_test, r2_test, rmse_test, pearson_test) = evaluate_model(
                pipeline, X_train, y_train, X_val, y_val, X_test, y_test, StandardScaler()
            )
            
            result = {
                'Versao': versao_nome,
                'Treinamento': f'{nome_train1}, {nome_train2}',
                'Validacao': nome_validacao,
                'Teste': nome_teste,
                'Model': model_name,
                'MAE_Val': mae_val,
                'R2_Val': r2_val,
                'RMSE_Val': rmse_val,
                'Pearson_Val': pearson_val,
                'MAE_Test': mae_test,
                'R2_Test': r2_test,
                'RMSE_Test': rmse_test,
                'Pearson_Test': pearson_test,
                'Execution_DateTime': datetime.now(),
                'Best_Params': str(best_params)
            }
            temp_results.append(result)
        
        # Ao terminar todos os ciclos para o modelo, exporta os resultados deste modelo
        resultados_df_model = pd.DataFrame(temp_results)
        resultados_df_model.drop_duplicates(inplace=True)
        preprocessing.exportar_resultados_para_csv(resultados_df_model, nome_arquivo_csv)
        print(f"Resultados exportados para o modelo: {model_name}")
        resultados_completos.extend(temp_results)
    
    print("Processamento finalizado!")
    return pd.DataFrame(resultados_completos)

def avaliar_modelos_em_datasets(lista_datasets, versao_nome, nome_arquivo_csv):
    """
    Avalia os modelos usando KFold.
    Para cada dataset e para cada modelo, divide os dados (usando os folds previamente salvos),
    separa internamente cada fold em treinamento e validação e depois re-treina com a junção
    dos dados de treinamento e validação para testar no conjunto de teste do fold.
    
    Os resultados (médias e desvios) de cada modelo são exportados uma única vez, por dataset.
    """
    resultados_completos = []
    models = get_models()
    
    with tqdm(total=len(lista_datasets), desc="Processando Datasets") as dataset_bar:
        for i, dados_filtrados in enumerate(lista_datasets):
            dataset_name = dados_filtrados['dataset_name'].iloc[0]
            print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')
            
            scaler = StandardScaler()
            normalize_data(scaler, [dados_filtrados])
            
            descriptions = np.array(dados_filtrados['treated_description'].values)
            effort_estimations = np.array(dados_filtrados['storypoint_scaled'].values)
            
            utils.salvar_kfold(dados_filtrados, dataset_name)
            kfold_indices = utils.carregar_kfold(dataset_name)
            
            # Para cada modelo, acumula os resultados de cada fold
            for (model_name, model) in models:
                print(f"\nIniciando o modelo: {model_name} para o dataset: {dataset_name}")
                
                mae_val_list = []
                r2_val_list = []
                rmse_val_list = []
                pearson_val_list = []
                
                mae_test_list = []
                r2_test_list = []
                rmse_test_list = []
                pearson_test_list = []
                
                # Para os modelos que usam Optuna, coletamos os melhores hiperparâmetros (opcional)
                best_params_model = 'N/A'
                
                # Processa cada fold
                with tqdm(total=len(kfold_indices), desc=f"Processando Folds para {model_name}") as fold_bar:
                    for train_index, test_index in kfold_indices:
                        X_fold_train = descriptions[train_index]
                        y_fold_train = effort_estimations[train_index]
                        X_fold_test = descriptions[test_index]
                        y_fold_test = effort_estimations[test_index]
                        
                        # Divisão interna do fold em treinamento e validação
                        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                            X_fold_train, y_fold_train, test_size=0.2, random_state=42
                        )
                        
                        # Se o modelo usa Optuna, otimize os hiperparâmetros com os dados internos
                        if isinstance(model, str):
                            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
                            study.optimize(lambda trial: objective(trial, X_train_inner, y_train_inner, X_val_inner, y_val_inner, model),
                                           n_trials=5, n_jobs=-1)
                            best_params_model = study.best_params
                            print(f"Melhores hiperparâmetros para {model_name} neste fold: {best_params_model}")
                            if model == 'Random Forest':
                                tuned_model = RandomForestRegressor(**best_params_model, n_jobs=-1, random_state=42)
                            elif model == 'XGBoost':
                                tuned_model = XGBRegressor(**best_params_model, random_state=42)
                            elif model == 'LightGBM':
                                tuned_model = LGBMRegressor(**best_params_model, random_state=42)
                            else:
                                tuned_model = None
                        else:
                            tuned_model = model
                        
                        pipeline = create_pipeline(tuned_model)
                        (mae_val, r2_val, rmse_val, pearson_val,
                         mae_test, r2_test, rmse_test, pearson_test) = evaluate_model(
                            pipeline, X_train_inner, y_train_inner, X_val_inner, y_val_inner, X_fold_test, y_fold_test, StandardScaler()
                        )
                        
                        mae_val_list.append(mae_val)
                        r2_val_list.append(r2_val)
                        rmse_val_list.append(rmse_val)
                        pearson_val_list.append(pearson_val)
                        
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
                    'MAE_Val_Mean': np.mean(mae_val_list),
                    'MAE_Val_Std': np.std(mae_val_list),
                    'R2_Val_Mean': np.mean(r2_val_list),
                    'R2_Val_Std': np.std(r2_val_list),
                    'RMSE_Val_Mean': np.mean(rmse_val_list),
                    'RMSE_Val_Std': np.std(rmse_val_list),
                    'Pearson_Val_Mean': np.mean(pearson_val_list),
                    'Pearson_Val_Std': np.std(pearson_val_list),
                    'MAE_Test_Mean': np.mean(mae_test_list),
                    'MAE_Test_Std': np.std(mae_test_list),
                    'R2_Test_Mean': np.mean(r2_test_list),
                    'R2_Test_Std': np.std(r2_test_list),
                    'RMSE_Test_Mean': np.mean(rmse_test_list),
                    'RMSE_Test_Std': np.std(rmse_test_list),
                    'Pearson_Test_Mean': np.mean(pearson_test_list),
                    'Pearson_Test_Std': np.std(pearson_test_list),
                    'Best_Params': str(best_params_model),
                    'Execution_DateTime': datetime.now()
                }
                
                # Exporta os resultados para este modelo (por este dataset) uma única vez
                resultados_df_model = pd.DataFrame([result])
                resultados_df_model.drop_duplicates(inplace=True)
                preprocessing.exportar_resultados_para_csv(resultados_df_model, nome_arquivo_csv)
                resultados_completos.append(result)
            
            dataset_bar.update(1)
    
    print("Processamento finalizado!")
    return pd.DataFrame(resultados_completos)