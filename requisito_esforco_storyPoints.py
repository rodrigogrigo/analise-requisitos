# %pip install langdetect
# %pip install pandas
# %pip install spacy
# %pip install scikit-learn
# %pip install bs4
# %pip install nltk
# %pip install xgboost
# %pip install catboost
# %pip install transformers torch

from langdetect import detect as detect_language, LangDetectException
from sklearn.base import BaseEstimator, RegressorMixin
from transformers import RobertaTokenizer, RobertaModel
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer
import nltk
import re
from bs4 import BeautifulSoup
from unicodedata import normalize
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import glob
import os
from langdetect import detect
import pandas as pd
import spacy

# Carregar o modelo de idioma inglês
nlp = spacy.load("en_core_web_sm")

# Função para detectar o idioma usando o Spacy


def detect_language(text):
    doc = nlp(text)
    return doc.lang_


def contar_frases(text):
    # Verifica se o texto não é nulo e contém conteúdo significativo
    if text is not None and len(text.strip()) > 0:
        doc = nlp(text)
        return len(list(doc.sents))
    else:
        return 0  # Retorna 0 se o texto for inválido ou vazio

# Função para contar o número de palavras em uma descrição usando spaCy


def contar_palavras(text):
    # Verifica se o texto não é nulo e contém conteúdo significativo
    if text is not None and len(text.strip()) > 0:
        doc = nlp(text)
        return len(list(doc))
    else:
        return 0  # Retorna 0 se o texto for inválido ou vazio


def detectar_idioma(texto):
    try:
        return detect_language(texto)
    except LangDetectException:
        return 'unknown'


def carregar_dados(fonte_dados):
    # Lendo Dados
    dados_gerais = pd.read_csv(fonte_dados)

    # Aplicar a função de detecção de idioma à coluna 'description' e armazenar o resultado em uma nova coluna
    dados_gerais['lang'] = dados_gerais['description'].apply(
        lambda x: detectar_idioma(str(x)))

    # Filtrar apenas os registros onde o idioma detectado é inglês
    dados = dados_gerais[dados_gerais['lang'] == 'en']

    # Filtrar registros com storypoint maior que zero e não nulo
    dados = dados[dados['storypoint'].notnull() & (dados['storypoint'] > 0)]

    # Selecionar apenas as colunas 'description', 'storypoint' e 'issuekey'
    dados_filtrados = dados[['description', 'storypoint', 'issuekey']]

    return dados_filtrados


def carregar_todos_dados(diretorio_pasta):
    lista_datasets = []
    # Encontrar todos os arquivos .csv no diretório especificado
    arquivos_csv = glob.glob(os.path.join(diretorio_pasta, "*.csv"))

    for arquivo in arquivos_csv:
        dados_filtrados = carregar_dados(arquivo)
        nome_dataset = os.path.basename(arquivo)  # Obtendo o nome do arquivo
        # Adicionando uma nova coluna com o nome do dataset
        dados_filtrados['dataset_name'] = nome_dataset
        lista_datasets.append(dados_filtrados)

    return lista_datasets


diretorio = 'D:\\Mestrado\\Python\\Projeto\\Datasets\\JIRA-Estimation-Prediction\\storypoint\\IEEE TSE2018\\dataset'
datasets = carregar_todos_dados(diretorio)

# Ver o primeiro DataFrame da lista
datasets[0].head()


def preprocessar_descricao(list_descricao):
    descricao_2 = []

    with tqdm(total=len(list_descricao), desc='Processando Descrição') as pbar:
        for descricao in list_descricao:
            descricao_processada = BeautifulSoup(
                descricao, 'html.parser').get_text()
            doc = nlp(descricao_processada)
            tokens = [t.lemma_.lower() for t in doc if t.pos_ != 'PUNCT'
                      and len(t.lemma_) > 1 and not t.is_stop]
            descricao_processada = ' '.join(tokens).strip()
            descricao_2.append(descricao_processada)
            pbar.update(1)

    return descricao_2


def preprocessar_todos_datasets(lista_datasets):
    for dados_filtrados in lista_datasets:
        dados_filtrados['treated_description'] = preprocessar_descricao(
            dados_filtrados['description'].values)

    return lista_datasets


# Exemplo de uso
datasets = preprocessar_todos_datasets(datasets)

# Verificar a nova coluna no primeiro dataset
datasets[0].head()


def analisar_datasets(lista_datasets):
    for i, dados_filtrados in enumerate(lista_datasets):
        # Obtendo o nome do dataset
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values

        min_effort = min(effort_estimations)
        max_effort = max(effort_estimations)

        effort_estimations = np.array(effort_estimations)

        print(
            f'\nTotal Examples: {len(descriptions)} -- {len(effort_estimations)}\n')
        print(f'\nMin Estimation: {min_effort}')
        print(f'Max Estimation: {max_effort}')

        # Boxplot dos valores de esforço
        plt.figure(figsize=(10, 6))
        plt.boxplot(effort_estimations)
        plt.title(f'{dataset_name}: Boxplot dos Valores de Esforço')
        plt.ylabel('Esforço (em unidades de tempo)')
        plt.show()

        # Calcular os quartis e o IQR
        Q1 = np.percentile(effort_estimations, 25)
        Q3 = np.percentile(effort_estimations, 75)
        IQR = Q3 - Q1

        # Determinar os limites inferior e superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identificar outliers
        outliers = effort_estimations[(effort_estimations < lower_bound) | (
            effort_estimations > upper_bound)]

        print(f"Outliers:\n{outliers}")
        print(f"Lower Bound: {lower_bound}")
        print(f"Upper Bound: {upper_bound}")


# Exemplo de uso
analisar_datasets(datasets)


def remover_outliers_e_filtrar(lista_datasets):
    nova_lista_datasets = []

    for i, dados_filtrados in enumerate(lista_datasets):
        # Obtendo o nome do dataset
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nProcessando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values

        # Calcular os quartis e o IQR
        Q1 = np.percentile(effort_estimations, 25)
        Q3 = np.percentile(effort_estimations, 75)
        IQR = Q3 - Q1

        # Determinar os limites inferior e superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remover os outliers
        mask = (effort_estimations >= lower_bound) & (
            effort_estimations <= upper_bound)
        filtered_effort_estimations = effort_estimations[mask]
        filtered_descriptions = descriptions[mask]

        print(
            f'\nTotal Examples After Removing Outliers: {len(filtered_descriptions)} -- {len(filtered_effort_estimations)}\n')
        print(
            f'\nTotal Examples Removed: {len(effort_estimations) - len(filtered_effort_estimations)}\n')

        # Criar um novo DataFrame sem os outliers
        dados_filtrados_sem_outliers = dados_filtrados[mask].copy()

        # Adicionar o DataFrame sem outliers à nova lista
        nova_lista_datasets.append(dados_filtrados_sem_outliers)

        # Boxplot dos valores de esforço sem outliers
        plt.figure(figsize=(10, 6))
        plt.boxplot(filtered_effort_estimations)
        plt.title(
            f'{dataset_name}: Boxplot dos Valores de Esforço sem Outliers')
        plt.ylabel('Esforço (em unidades de tempo)')
        plt.show()

    return nova_lista_datasets


# Exemplo de uso
datasets_sem_outliers = remover_outliers_e_filtrar(datasets)


def avaliar_modelos_em_datasets(lista_datasets):
    models = [
        ('Linear Regression', LinearRegression(), None),
        ('Ridge Regression', Ridge(), {
         'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
        ('Lasso Regression', Lasso(max_iter=100000, tol=0.01), {
         'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}),
        ('ElasticNet Regression', ElasticNet(max_iter=100000), {'regressor__alpha': [
         0.1, 1.0, 10.0, 100.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
        ('SVR', SVR(), {'regressor__C': [
         0.1, 1.0, 10.0], 'regressor__epsilon': [0.01, 0.1, 1.0]}),
        ('KNN', KNeighborsRegressor(), {
         'regressor__n_neighbors': [3, 5, 7, 9]}),
        ('Decision Tree', DecisionTreeRegressor(), {
         'regressor__max_depth': [None, 10, 20, 30]}),
        ('XGBoost', XGBRegressor(), {'regressor__n_estimators': [
         100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.3]}),
        ('CatBoost', CatBoostRegressor(silent=True), {'regressor__iterations': [
         100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.3]}),
    ]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=1_000)),
        ('normalizer', Normalizer()),  # Normalizar os dados
        ('regressor', None)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    resultados_completos = []

    for i, dados_filtrados in enumerate(lista_datasets):
        # Obtendo o nome do dataset
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values

        results = {
            'Dataset': [],
            'Model': [],
            'MAE': [],
            'R2': [],
            'RMSE': [],
            'Pearson Correlation': []
        }

        for name, model, param_grid in models:
            pipeline.set_params(regressor=model)
            print(f"\nModel: {name}")
            regressor = None
            if param_grid:
                regressor = GridSearchCV(
                    pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
            else:
                regressor = pipeline

            list_maes_scores = []
            list_r2_scores = []
            list_rmse_scores = []
            list_pearson_scores = []

            for train_index, test_index in kf.split(descriptions):
                X_train, X_test = descriptions[train_index], descriptions[test_index]
                y_train, y_test = effort_estimations[train_index], effort_estimations[test_index]
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

            results['Dataset'].append(dataset_name)
            results['Model'].append(name)
            results['MAE'].append(
                f"{np.mean(list_maes_scores)} ± {np.std(list_maes_scores)}")
            results['R2'].append(
                f"{np.mean(list_r2_scores)} ± {np.std(list_r2_scores)}")
            results['RMSE'].append(
                f"{np.mean(list_rmse_scores)} ± {np.std(list_rmse_scores)}")
            results['Pearson Correlation'].append(
                f"{np.mean(list_pearson_scores)} ± {np.std(list_pearson_scores)}")

        # Adiciona os resultados do dataset atual à lista completa de resultados
        resultados_completos.append(pd.DataFrame(results))

    # Combina todos os resultados em um único DataFrame
    resultados_finais = pd.concat(resultados_completos, ignore_index=True)

    return resultados_finais


# Exemplo de uso
resultados_finais = avaliar_modelos_em_datasets(datasets_sem_outliers)

# Mostrar os resultados
print("\nResultados Completos:")
resultados_finais.head()


# Carregar tokenizer e modelo do CodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# Função para gerar embeddings usando CodeBERT em batches


def get_codebert_embeddings_in_batches(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt',
                           padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

# Função para processar todos os datasets


def processar_datasets_com_codebert(lista_datasets):
    resultados_completos = []

    for i, dados_filtrados in enumerate(lista_datasets):
        # Obtendo o nome do dataset
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nProcessando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values

        # Transformar as descrições das tarefas em embeddings
        codebert_embeddings = get_codebert_embeddings_in_batches(
            descriptions.tolist(), tokenizer, model, batch_size=16)

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            codebert_embeddings, effort_estimations, test_size=0.2, random_state=42)

        # Treinar o modelo Ridge Regression
        ridge = Ridge()
        ridge.fit(X_train, y_train)

        # Fazer previsões
        y_pred = ridge.predict(X_test)

        # Avaliar o modelo
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")

        # Adicionar resultados à lista de resultados completos
        new_results = pd.DataFrame({
            'Dataset': [dataset_name],
            'Model': ['CodeBERT + Ridge'],
            'MAE': [mae],
            'R2 Score': [r2],
            'MSE': [mse],
            'RMSE': [rmse]
        })

        resultados_completos.append(new_results)

    # Combina todos os resultados em um único DataFrame
    resultados_finais = pd.concat(resultados_completos, ignore_index=True)

    return resultados_finais


# Exemplo de uso
resultados_finais = processar_datasets_com_codebert(datasets_sem_outliers)

# Mostrar os resultados finais
print("\nResultados Completos:")
resultados_finais.head()
