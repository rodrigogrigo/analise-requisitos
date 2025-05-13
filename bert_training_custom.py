#!pip uninstall -y transformers tokenizers
#!rm -rf /usr/local/lib/python*/dist-packages/transformers*
#!rm -rf /usr/local/lib/python*/dist-packages/tokenizers*
#!pip install transformers==4.37.2 datasets
#
#!pip install peft==0.10.0
#
#!pip install accelerate==0.26.1

import transformers
print(transformers.__version__)

import os
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
)
import torch

# Caminhos
PASTA_DADOS = "treinamento_model"
MODELO_BASE = "bert-base-uncased"
PASTA_SAIDA_MODELO = "models/custom/bert_treinado_custom"

# 1. Carregar os CSVs
def carregar_datasets_da_pasta(pasta):
    arquivos_csv = [f for f in os.listdir(pasta) if f.endswith('.csv')]
    datasets = []
    for arquivo in arquivos_csv:
        df = pd.read_csv(os.path.join(pasta, arquivo))
        if 'treated_description_bert' in df.columns:
            ds = Dataset.from_pandas(df[['treated_description_bert']].rename(columns={'treated_description_bert': 'text'}))
            datasets.append(ds)
    return concatenate_datasets(datasets)

print("Carregando dados...")
dataset_total = carregar_datasets_da_pasta(PASTA_DADOS)

# 2. Tokenização
print("Tokenizando textos...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized_dataset = dataset_total.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Dividir em treino e validação (90% treino, 10% validação)
print("Separando treino e validação...")
train_test_split_result = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split_result["train"]
val_dataset = train_test_split_result["test"]

# 4. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# 5. Modelo base
model = AutoModelForMaskedLM.from_pretrained(MODELO_BASE)

# 6. Configuração de treinamento
training_args = TrainingArguments(
    output_dir=PASTA_SAIDA_MODELO,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    #per_device_train_batch_size=16,
    per_device_train_batch_size=4,
    #num_train_epochs=3,
    num_train_epochs=1,
    #save_steps=10_000,
    save_total_limit=1,
    push_to_hub=False,
    logging_steps=500,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. Treinamento
print("Iniciando treinamento...")
trainer.train()

# 9. Salvamento
print(f"Salvando modelo em {PASTA_SAIDA_MODELO}")
model.save_pretrained(PASTA_SAIDA_MODELO)
tokenizer.save_pretrained(PASTA_SAIDA_MODELO)
