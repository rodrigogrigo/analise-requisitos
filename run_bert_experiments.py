from preprocessing import carregar_datasets_unificado
from bert_evaluation import avaliar_modelo_bert_intra_datasets, avaliar_modelo_bert_inter_datasets


if __name__ == '__main__':

    versao_abordagem = 'V_BERT'
    diretorio_datasets_brutos = 'datasets_all/'
    diretorio_datasets_processados = 'datasets_all_processados/'

    limitar_qtde_registros = False
    n_registros = -1

    is_run_intra_datasets = True

    bert_datasets = carregar_datasets_unificado(
        diretorio_datasets_brutos,
        diretorio_datasets_processados,
        limitar_qtde_registros,
        n_registros)

    if is_run_intra_datasets:

        avaliar_modelo_bert_intra_datasets(
            bert_datasets,
            versao_abordagem,
            nome_arquivo_resultados='resultados_modelosBERT_intra_dataset_2.csv'
        )

    else:

        avaliar_modelo_bert_inter_datasets(
            bert_datasets,
            versao_abordagem,
            nome_arquivo_resultados='resultados_modelosBERT_inter_dataset_2.csv'
        )
