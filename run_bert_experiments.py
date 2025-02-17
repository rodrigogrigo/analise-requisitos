from preprocessing import carregar_todos_dados
from bert_evaluation import avaliar_modelo_bert_intra_datasets


if __name__ == '__main__':

    VERSAO_NOME_BERT = 'V_BERT'
    DIRETORIO_DATASET_BRUTO = 'datasets_all/'
    DIRETORIO_DATASET_PROCESSADO = 'datasets_all_processados/'
    LIMITAR_QUANTIDADE_REGISTROS = False
    QUANTIDADE_REGISTROS_SE_LIMITADO = -1
    NOME_ARQUIVO_RESULTADOS = 'resultados_modelos.csv'

    _, bert_datasets = carregar_todos_dados(
        DIRETORIO_DATASET_BRUTO, DIRETORIO_DATASET_PROCESSADO, LIMITAR_QUANTIDADE_REGISTROS,
        QUANTIDADE_REGISTROS_SE_LIMITADO)

    resultados_finais, predicoes_por_modelo = avaliar_modelo_bert_intra_datasets(
        bert_datasets, VERSAO_NOME_BERT, NOME_ARQUIVO_RESULTADOS
    )
