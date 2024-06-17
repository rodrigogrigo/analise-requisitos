import numpy as np
import spacy
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error


def detect_language(text):
    doc = nlp(text)
    return doc.lang_


"""
    1. Rever a função de pré-processamento. Tem coisas lá que está sendo feita, como remover stopwords manualmente,
        que já é feito pelo spacy.        
        
        1.1 Simplificar a função para usar só o Spacy. Não colocar remoção de rare ou common words.
        1.2 O pipeline deve ser:
            a) remover as tags html.
            b) usar o spacy para processar o texto e remover pontuação e stopwords.
"""

if __name__ == '__main__':

    dataset_path = 'data/datasets/JiraRepos.JiraEcosystem.csv'

    nlp = spacy.load('en_core_web_sm')

    dataset_df = pd.read_csv(dataset_path)

    dataset_df['lang'] = dataset_df['fields.description'].apply(lambda x: detect_language(str(x)))

    dataset_df = dataset_df[dataset_df['lang'] == 'en']

    dataset_df = dataset_df[['fields.description', 'fields.timeestimate', 'id']]

    descriptions = dataset_df['fields.description'].values
    effort_estimations = dataset_df['fields.timeestimate'].values

    min_effort = min(effort_estimations)
    max_effort = max(effort_estimations)

    effort_estimations = np.array(effort_estimations)

    print(f'\nTotal Examples: {len(descriptions)} -- {len(effort_estimations)}\n')

    print(f'\nMin Estimation: {min_effort}')
    print(f'Max Estimation: {max_effort}')

    models = [
        ('Linear Regression', LinearRegression(), {}),
        ('Ridge Regression', Ridge(), {
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
        ('Lasso Regression', Lasso(max_iter=10000), {
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
        ('ElasticNet Regression', ElasticNet(max_iter=10000), {'regressor__alpha': [
            0.1, 1.0, 10.0, 100.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
        ('SVR', SVR(), {'regressor__C': [
            0.1, 1.0, 10.0], 'regressor__epsilon': [0.01, 0.1, 1.0]}),
        ('KNN', KNeighborsRegressor(), {'regressor__n_neighbors': [3, 5, 7, 9]}),
        ('Decision Tree', DecisionTreeRegressor(), {
            'regressor__max_depth': [None, 10, 20, 30]})
    ]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=1_000)),
        ('regressor', None)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model, param_grid in models:

        pipeline.set_params(regressor=model)

        print(f"\nModel: {name}")

        regressor = None

        if param_grid:
            regressor = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            regressor = pipeline

        list_maes_scores = []

        for train_index, test_index in kf.split(descriptions):

            X_train, X_test = descriptions[train_index], descriptions[test_index]

            y_train, y_test = effort_estimations[train_index], effort_estimations[test_index]

            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)

            list_maes_scores.append(mae)

        print(f'\n  MAE: {np.mean(list_maes_scores)} ~ {np.std(list_maes_scores)}')
