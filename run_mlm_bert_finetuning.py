import  os
import pandas as pd
import math

from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


def carregar_datasets_da_pasta(pasta: str):
    arquivos_csv = [f for f in os.listdir(pasta) if f.endswith('.csv')]
    datasets = []
    for arquivo in arquivos_csv:
        df = pd.read_csv(os.path.join(pasta, arquivo))
        if 'treated_description_bert' in df.columns:
            ds = Dataset.from_pandas(
                df[['treated_description_bert']].rename(columns={'treated_description_bert': 'text'})
            )
            datasets.append(ds)
    return concatenate_datasets(datasets)


def tokenize_function(examples, tokenizer_):
    return tokenizer_(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


if __name__ == '__main__':

    PASTA_DADOS = 'treinamento_model'

    model_tuple = ("mlm_bert_base_uncased", "google-bert/bert-base-uncased")
    # model_tuple = ("mlm_roberta_base", "FacebookAI/roberta-base")
    # model_tuple = ("mlm_code_bert", "microsoft/codebert-base")
    # model_tuple = ("mlm_se_bert", "thearod5/se-bert")
    # model_tuple = ("mlm_bert_software_engineering", "burakkececi/bert-software-engineering")
    # model_tuple = ("mlm_bert_large_uncased", "google-bert/bert-large-uncased")
    # model_tuple = ("mlm_roberta_large", "FacebookAI/roberta-large")

    print(f'\nModel: {model_tuple[0]}')

    PASTA_SAIDA_MODELO = f'models/finetuned_mlm/{model_tuple[0]}'

    num_epochs = 50
    batch_size = 16

    print("\nCarregando dados...")

    dataset_total = carregar_datasets_da_pasta(PASTA_DADOS)

    print("\nTokenizando textos...")

    tokenizer = AutoTokenizer.from_pretrained(model_tuple[1])

    tokenized_dataset = dataset_total.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        fn_kwargs={
            'tokenizer_': tokenizer
        }
    )

    print("\nSeparando treino e validação...")

    train_test_split_result = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = train_test_split_result["train"]

    val_dataset = train_test_split_result["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    print(f'\nTotal Train: {len(train_dataset)}')
    print(f'Total Validation: {len(val_dataset)}')

    model = AutoModelForMaskedLM.from_pretrained(model_tuple[1])

    training_args = TrainingArguments(
        output_dir=PASTA_SAIDA_MODELO,
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_total_limit=2,
        push_to_hub=False,
        logging_steps=500,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\nIniciando treinamento...")

    if os.path.exists(PASTA_SAIDA_MODELO) and len(os.listdir(PASTA_SAIDA_MODELO)) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    eval_results = trainer.evaluate()

    print(f"\nPerplexity: {math.exp(eval_results['eval_loss']):.2f}")

    BEST_MODEL_PATH = os.path.join(PASTA_SAIDA_MODELO, 'best_model')

    print(f"\nSalvando modelo em {PASTA_SAIDA_MODELO}")

    trainer.save_model(BEST_MODEL_PATH)

    tokenizer.save_pretrained(BEST_MODEL_PATH)
