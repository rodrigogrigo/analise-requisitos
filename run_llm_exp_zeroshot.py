import os
import torch
import pandas as pd

from unsloth import FastLanguageModel


if __name__ == '__main__':

    prompt_template_path = 'prompt_llm_zeroshot.txt'

    datasets_dir = 'datasets_all_processados'

    # llm_tuple = ('gemma2_9b', 'unsloth/unsloth/gemma-2-9b-it-bnb-4bit')
    # llm_tuple = ('gemma3_12b', 'unsloth/gemma-3-12b-it-unsloth-bnb-4bit')
    # llm_tuple = ('llama31_8b', 'unsloth/Llama-3.1-8B-unsloth-bnb-4bit')
    # llm_tuple = ('qwen3_8B', 'unsloth/Qwen3-8B-unsloth-bnb-4bit')
    llm_tuple = ('Qwen25_coder_14B', 'unsloth/Qwen2.5-Coder-14B-bnb-4bit')

    with open(file=prompt_template_path, mode='r') as file:
        prompt_template = file.read()

    list_datasets_names = os.listdir(datasets_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'\nDevice: {device}')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=llm_tuple[1],
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    print(f'\n{40*"="} Running Experiment LLM ZeroShot {40*"="}')

    for dataset_name in list_datasets_names:

        print(f'\n\tDataset: {dataset_name}')

        dataset_path = os.path.join(datasets_dir, dataset_name)

        dataset_df = pd.read_csv(dataset_path)

        for _, row in dataset_df.iterrows():

            requirement_description = row['description']
            real_storypoint = row['storypoint']

            full_prompt = prompt_template.format(requirement_description=requirement_description)

            inputs = tokenizer(
                full_prompt,
                truncation=True,
                max_length=2048,
                padding=True,
                return_tensors='pt',
            ).to(device)

            print(inputs)

            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=8,
                temperature=0.1,
            )

            print(outputs)

            generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            print(generated_output)

            break

        break






