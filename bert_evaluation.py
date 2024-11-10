from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
from datetime import datetime
import utils
import importlib
importlib.reload(utils)

LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20


class StoryPointDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze(-1)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    pearson_corr, _ = pearsonr(labels, preds)
    return {
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'pearson_corr': pearson_corr
    }


def avaliar_modelo_bert_em_datasets(lista_datasets, versao_nome):
    # Lista de modelos a serem utilizados
    modelos = ['bert-base-uncased',
               'microsoft/codebert-base', 'FacebookAI/roberta-base']
    resultados_completos = []
    predicoes_por_modelo = {}

    # Loop pelos modelos de BERT para avaliação
    for model_name in modelos:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1)

        # Loop para cada dataset
        for i, dados_filtrados in enumerate(lista_datasets):
            dataset_name = dados_filtrados['dataset_name'].iloc[0]
            print(
                f'\nAnalisando Dataset {i + 1} com o modelo {model_name}: {dataset_name}\n')

            descriptions = dados_filtrados['treated_description'].values
            story_points = dados_filtrados['storypoint'].values
            issuekeys = dados_filtrados['issuekey'].values

            # Salva e carrega os índices de KFold para garantir a mesma divisão
            utils.salvar_kfold(descriptions, dataset_name)
            kfold_indices = utils.carregar_kfold(dataset_name)

            results = {
                'Versao': [], 'Dataset': [], 'Model': [], 'MAE_Mean': [], 'MAE_Std': [],
                'R2_Mean': [], 'R2_Std': [], 'RMSE_Mean': [], 'RMSE_Std': [],
                'Pearson_Corr_Mean': [], 'Pearson_Corr_Std': [], 'Execution_DateTime': []
            }

            # Listas para armazenar as métricas de cada fold
            list_maes_scores, list_r2_scores, list_rmse_scores, list_pearson_scores = [], [], [], []
            all_y_test, all_y_pred, all_descriptions_test, all_issuekeys_test = [], [], [], []

            # Loop para cada fold nos índices carregados
            for train_index, test_index in kfold_indices:
                # Cria conjuntos de treino e teste a partir dos índices
                X_train_full, X_test = descriptions[train_index], descriptions[test_index]
                y_train_full, y_test = story_points[train_index], story_points[test_index]
                issuekeys_test = issuekeys[test_index]

                # Divide o conjunto de treino em treino e validação (10% para validação)
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X_train_full, y_train_full, test_size=0.1, random_state=42
                )

                # Cria os datasets para treino, validação e teste
                train_dataset = StoryPointDataset(
                    X_train, y_train, tokenizer, MAX_LENGTH)
                valid_dataset = StoryPointDataset(
                    X_valid, y_valid, tokenizer, MAX_LENGTH)
                test_dataset = StoryPointDataset(
                    X_test, y_test, tokenizer, MAX_LENGTH)

                # Configura os argumentos de treinamento
                training_args = TrainingArguments(
                    output_dir='./results',
                    learning_rate=LEARNING_RATE,
                    per_device_train_batch_size=BATCH_SIZE,
                    per_device_eval_batch_size=BATCH_SIZE,
                    num_train_epochs=EPOCHS,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=2,
                    metric_for_best_model="mae",
                    load_best_model_at_end=True,
                    weight_decay=0.01,
                )

                # Inicializa o Trainer com datasets de treino e validação
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=valid_dataset,  # Usa o conjunto de validação
                    compute_metrics=compute_metrics
                )

                # Treinamento e avaliação
                trainer.train()
                eval_results = trainer.evaluate()

                # Armazena as métricas para cada fold
                list_maes_scores.append(eval_results['eval_mae'])
                list_r2_scores.append(eval_results['eval_r2'])
                list_rmse_scores.append(eval_results['eval_rmse'])
                list_pearson_scores.append(eval_results['eval_pearson_corr'])

                # Realiza as predições no conjunto de teste
                predictions = trainer.predict(
                    test_dataset).predictions.flatten()
                all_y_test.extend(y_test)
                all_y_pred.extend(predictions)
                all_descriptions_test.extend(X_test)
                all_issuekeys_test.extend(issuekeys_test)

            # Salva as predições e métricas de cada modelo
            predicoes_por_modelo[model_name] = {
                'issuekeys': all_issuekeys_test,
                'descriptions': all_descriptions_test,
                'y_test': all_y_test,
                'y_pred': all_y_pred
            }

            # Atualiza os resultados agregados do modelo e dataset
            results['Versao'].append(versao_nome)
            results['Dataset'].append(dataset_name)
            results['Model'].append(model_name)
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

    # Concatena todos os resultados e retorna
    resultados_finais = pd.concat(resultados_completos, ignore_index=True)
    return resultados_finais, predicoes_por_modelo
