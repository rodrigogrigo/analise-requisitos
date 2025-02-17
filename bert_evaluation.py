import torch
import numpy as np
import pandas as pd
import utils
import importlib
import preprocessing
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
from datetime import datetime

importlib.reload(utils)
importlib.reload(preprocessing)

LEARNING_RATE = 5e-5
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 1
BASE_CHECKPOINT_PATH = 'models/'  # Para Colab, use "/content/checkpoints"


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


import shutil

def get_models():
    return [
        ("bert_base_uncased", "google-bert/bert-base-uncased"),
        ("roberta_base", "FacebookAI/roberta-base"),
        ("code_bert", "microsoft/codebert-base"),
        ("se_bert", "thearod5/se-bert"),
        ("bert_software_engineering", "burakkececi/bert-software-engineering"),
        ("bert_large_uncased", "google-bert/bert-large-uncased"),
        ("roberta_large", "FacebookAI/roberta-large")
    ]

def avaliar_modelo_bert_intra_datasets(lista_datasets, versao_nome, nome_arquiv_resultados):

    modelos = get_models()

    resultados_completos = []

    predicoes_por_modelo = {}

    for model_name, model_checkpoint in modelos:

        print(f"\nProcessando Modelo {model_name}")

        base_model_path = os.path.join(BASE_CHECKPOINT_PATH, model_name)

        os.makedirs(base_model_path, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

        for i, dados_filtrados in enumerate(lista_datasets):

            dataset_name = dados_filtrados["dataset_name"].iloc[0]

            print(f"\n\tProcessando Dataset {dataset_name}\n")

            descriptions = dados_filtrados["treated_description"].values

            story_points = dados_filtrados["storypoint"].values

            issuekeys = dados_filtrados["issuekey"].values

            utils.salvar_kfold(dados_filtrados, dataset_name)

            kfold_indices = utils.carregar_kfold(dataset_name)

            # Inicializar listas para métricas e predições

            list_maes_scores, list_r2_scores, list_rmse_scores, list_pearson_scores = [], [], [], []

            all_y_test, all_y_pred, all_descriptions_test, all_issuekeys_test = [], [], [], []

            for fold_idx, (train_index, test_index) in enumerate(kfold_indices, start=1):

                print(f"\t\tProcessando Fold {fold_idx}")

                # print(f"  Processando Fold {fold_idx + 1}/{len(kfold_indices)} para o modelo {model_name}")

                X_train_full, X_test = descriptions[train_index], descriptions[test_index]
                y_train_full, y_test = story_points[train_index], story_points[test_index]

                fold_best_model_path = os.path.join(base_model_path,
                                               f"fold_{fold_idx}", "best_model")

                if os.path.exists(os.path.join(fold_best_model_path, "pytorch_model.bin")):

                    print(f"Carregando modelo pré-treinado para o fold {fold_idx}.")

                    model = AutoModelForSequenceClassification.from_pretrained(
                        fold_best_model_path, num_labels=1)

                else:

                    os.makedirs(fold_best_model_path, exist_ok=True)

                    # print(f"Treinando modelo para o fold {fold_idx}.")

                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_checkpoint, num_labels=1)

                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X_train_full, y_train_full, test_size=0.1, random_state=42
                    )

                    train_dataset = StoryPointDataset(X_train, y_train, tokenizer, MAX_LENGTH)

                    valid_dataset = StoryPointDataset(X_valid, y_valid, tokenizer, MAX_LENGTH)

                    output_dir = os.path.join(base_model_path, f"fold_{fold_idx}", "checkpoints")

                    os.makedirs(output_dir, exist_ok=True)

                    training_args = TrainingArguments(
                        output_dir=output_dir,
                        learning_rate=LEARNING_RATE,
                        per_device_train_batch_size=BATCH_SIZE,
                        per_device_eval_batch_size=BATCH_SIZE,
                        num_train_epochs=EPOCHS,
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        save_total_limit=1,
                        load_best_model_at_end=True,
                        fp16=False,
                        weight_decay=0.01,
                        metric_for_best_model="mae",
                        greater_is_better=False,
                        report_to="none",
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=valid_dataset,
                        compute_metrics=compute_metrics,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                    )

                    trainer.train()

                    model.save_pretrained(fold_best_model_path)

                    tokenizer.save_pretrained(fold_best_model_path)

                    shutil.rmtree(output_dir, ignore_errors=True)

                test_dataset = StoryPointDataset(X_test, y_test, tokenizer, MAX_LENGTH)

                trainer = Trainer(model=model)

                predictions = trainer.predict(test_dataset).predictions.flatten()

                # Calcular métricas
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                pearson_corr, _ = pearsonr(y_test, predictions)

                # Salvar métricas do fold
                list_maes_scores.append(mae)
                list_r2_scores.append(r2)
                list_rmse_scores.append(rmse)
                list_pearson_scores.append(pearson_corr)

                # Salvar predições do fold
                all_y_test.extend(y_test)
                all_y_pred.extend(predictions)
                all_descriptions_test.extend(X_test)
                all_issuekeys_test.extend(issuekeys[test_index])

            # Consolidar resultados para o modelo atual no dataset
            result = {
                "Versao": versao_nome,
                "Dataset": dataset_name,
                "Model": model_name,
                "MAE_Mean": np.mean(list_maes_scores),
                "MAE_Std": np.std(list_maes_scores),
                "R2_Mean": np.mean(list_r2_scores),
                "R2_Std": np.std(list_r2_scores),
                "RMSE_Mean": np.mean(list_rmse_scores),
                "RMSE_Std": np.std(list_rmse_scores),
                "Pearson_Corr_Mean": np.mean(list_pearson_scores),
                "Pearson_Corr_Std": np.std(list_pearson_scores),
                "Execution_DateTime": datetime.now(),
            }

            resultados_completos.append(result)

            # Exportar resultados do modelo consolidado para o CSV
            resultados_df = pd.DataFrame([result])
            resultados_df.drop_duplicates(inplace=True)
            preprocessing.exportar_resultados_para_csv(resultados_df, nome_arquiv_resultados)

        # Limpar checkpoints para o modelo atual
        #
        print(f"Todos os checkpoints do modelo {model_name} foram removidos após a avaliação.")


    return pd.DataFrame(resultados_completos), predicoes_por_modelo
def avaliar_modelo_bert_em_datasets_ciclico(lista_datasets, versao_nome, nome_arquiv_resultados):
    """
    Avalia modelos BERT utilizando a seguinte estratégia cíclica:
      - Dataset i e i+1: Treinamento (concatenados)
      - Dataset i+2: Validação
      - Dataset i+3: Teste
    Em cada ciclo, o resultado inclui as bases utilizadas para treinamento, validação e teste.
    """
    modelos = get_models()

    resultados_completos = []
    predicoes_por_modelo = {}

    n_datasets = len(lista_datasets)
    # Loop pelos modelos
    with tqdm(total=len(modelos), desc="Processando Modelos") as modelo_bar:
        for model_name in modelos:
            base_model_path = os.path.join(BASE_CHECKPOINT_PATH, model_name)
            os.makedirs(base_model_path, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
            # Para cada modelo, processa os ciclos (cada ciclo usa 4 datasets de forma cíclica)
            with tqdm(total=n_datasets, desc=f"Processando Ciclos para {model_name}") as ciclo_bar:
                for i in range(n_datasets):
                    # Seleciona os datasets de forma cíclica:
                    ds_train1 = lista_datasets[i]
                    ds_train2 = lista_datasets[(i + 1) % n_datasets]
                    ds_val    = lista_datasets[(i + 2) % n_datasets]
                    ds_test   = lista_datasets[(i + 3) % n_datasets]
                    
                    # Extração dos nomes dos datasets para log
                    nome_train1 = ds_train1["dataset_name"].iloc[0]
                    nome_train2 = ds_train2["dataset_name"].iloc[0]
                    nome_val    = ds_val["dataset_name"].iloc[0]
                    nome_test   = ds_test["dataset_name"].iloc[0]
                    
                    print(f"\nCiclo {i+1}/{n_datasets} para o modelo {model_name}")
                    print(f"  Treinamento: {nome_train1} e {nome_train2}")
                    print(f"  Validação:   {nome_val}")
                    print(f"  Teste:       {nome_test}")
                    
                    # Preparação dos dados:
                    # Concatena os dados dos dois conjuntos de treinamento
                    X_train = np.concatenate([
                        ds_train1["treated_description"].values,
                        ds_train2["treated_description"].values
                    ])
                    y_train = np.concatenate([
                        ds_train1["storypoint"].values,
                        ds_train2["storypoint"].values
                    ])
                    
                    X_val = ds_val["treated_description"].values
                    y_val = ds_val["storypoint"].values
                    
                    X_test = ds_test["treated_description"].values
                    y_test = ds_test["storypoint"].values
                    
                    cycle_model_path = os.path.join(base_model_path, f"cycle_{i}")
                    os.makedirs(cycle_model_path, exist_ok=True)
                    
                    # Se já existir um modelo treinado para este ciclo, carrega-o; caso contrário, treina
                    if os.path.exists(os.path.join(cycle_model_path, "pytorch_model.bin")):
                        print(f"Carregando modelo pré-treinado para o ciclo {i}.")
                        model = AutoModelForSequenceClassification.from_pretrained(cycle_model_path)
                    else:
                        print(f"Treinando modelo para o ciclo {i}.")
                        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
                        
                        # Cria os datasets para treinamento e validação
                        train_dataset = StoryPointDataset(X_train, y_train, tokenizer, MAX_LENGTH)
                        val_dataset = StoryPointDataset(X_val, y_val, tokenizer, MAX_LENGTH)
                        
                        training_args = TrainingArguments(
                            output_dir=cycle_model_path,
                            learning_rate=LEARNING_RATE,
                            per_device_train_batch_size=BATCH_SIZE,
                            per_device_eval_batch_size=BATCH_SIZE,
                            num_train_epochs=EPOCHS,
                            evaluation_strategy="epoch",
                            save_strategy="epoch",
                            save_total_limit=1,
                            load_best_model_at_end=True,
                            fp16=True,
                            weight_decay=0.01,
                            metric_for_best_model="mae",
                            greater_is_better=False,
                            report_to="none",
                        )
                        
                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                        )
                        
                        trainer.train()
                        model.save_pretrained(cycle_model_path)
                        tokenizer.save_pretrained(cycle_model_path)
                    
                    # Avaliação final no conjunto de teste
                    test_dataset = StoryPointDataset(X_test, y_test, tokenizer, MAX_LENGTH)
                    trainer = Trainer(model=model)
                    predictions = trainer.predict(test_dataset).predictions.flatten()
                    
                    # Cálculo das métricas para o ciclo atual
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    pearson_corr, _ = pearsonr(y_test, predictions)
                    
                    # Registra o resultado deste ciclo, incluindo os nomes das bases utilizadas
                    result = {
                        "Versao": versao_nome,
                        "Model": model_name,
                        "Dataset_Treino": f"{nome_train1}, {nome_train2}",
                        "Dataset_Validacao": nome_val,
                        "Dataset_Teste": nome_test,
                        "MAE": mae,
                        "R2": r2,
                        "RMSE": rmse,
                        "Pearson_Corr": pearson_corr,
                        "Execution_DateTime": datetime.now()
                    }
                    resultados_completos.append(result)
                    
                    # Exporta o resultado deste ciclo para o CSV
                    resultados_df = pd.DataFrame([result])
                    resultados_df.drop_duplicates(inplace=True)
                    preprocessing.exportar_resultados_para_csv(resultados_df, nome_arquiv_resultados)
                    
                    ciclo_bar.update(1)
            
            # Após avaliação, remove os checkpoints do modelo atual
            shutil.rmtree(base_model_path, ignore_errors=True)
            print(f"Todos os checkpoints do modelo {model_name} foram removidos após a avaliação.")
            
            modelo_bar.update(1)
    
    return pd.DataFrame(resultados_completos), predicoes_por_modelo