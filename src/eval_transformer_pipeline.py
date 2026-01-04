import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import split_text_3_4
from eval_lstm import generate_and_evaluate

model_name = 'distilgpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('Загрузка тестового датасета...')
test_data = pd.read_csv('../data/test.csv')
texts = test_data['tweet'].dropna().tolist()
selected_texts = random.sample(texts, 10)

prompts = []
references = []
for text in selected_texts:
    split_data = split_text_3_4(text, tokenizer)
    prompts.append(split_data['prompt'])
    references.append(split_data['target'])

results = generate_and_evaluate(model, tokenizer, prompts, references, device)
for key, value in results.items():
    print(f"{key}: {value:.4f}")
