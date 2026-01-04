import os
import random

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.data_utils import split_text_3_4
from src.eval_lstm import generate_and_evaluate
from src.lstm_model import SimpleLSTM
from src.lstm_train import train_model_with_rouge
from src.next_token_dataset import create_dataloader

if __name__ == '__main__':
    data_dir = '../data/'

    # Создаем директорию для чекпоинтов
    save_directory = 'checkpoints'
    os.makedirs(save_directory, exist_ok=True)

    # Загружаем токенизатор
    print('Загрузка токенизатора...')
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Добавляем специальные токены для начала/конца
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print('Загрузка тренировочного датасета...')
    train_data = pd.read_csv(f'{data_dir}train.csv')
    train_texts = (
        train_data['tweet'] if 'tweet' in train_data.columns
        else train_data.iloc[:, 0]
    )

    print('Загрузка валидационного датасета...')
    val_data = pd.read_csv(f'{data_dir}val.csv')
    val_texts = (
        val_data['tweet'] if 'tweet' in val_data.columns
        else val_data.iloc[:, 0]
    )

    print('Загрузка тестового датасета...')
    test_data = pd.read_csv(f'{data_dir}val.csv')
    texts = test_data['tweet'].dropna().tolist()
    selected_texts = random.sample(texts, 10)

    # Создаем Dataloader'ы
    print('Создание даталоадеров...')
    train_dataloader = create_dataloader(train_texts, tokenizer)
    val_dataloader = create_dataloader(val_texts, tokenizer)

    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Создаем модель
    vocab_size = len(tokenizer)
    model = SimpleLSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
    )

    print('\nМодель создана: ')
    print(f'  Параметров: {sum(p.numel() for p in model.parameters()):,}')
    print(f'  Размер словаря: {vocab_size}')

    # Обучаем модель
    trained_model, history = train_model_with_rouge(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        selected_texts=selected_texts,
        tokenizer=tokenizer,
        device=device,
        num_epochs=5,
        learning_rate=0.001,
        eval_every=50,  # Каждые 50 шагов
        save_dir=save_directory,
    )

    # Визуализируем историю обучения
    history.plot_training_history()

    # Финальная оценка
    print('\n' + '=' * 60)
    print('Финальная оценка модели:')
    print('=' * 60)

    prompts = []
    references = []
    for text in selected_texts:
        split_data = split_text_3_4(text, tokenizer)
        prompts.append(split_data['prompt'])
        references.append(split_data['target'])

    results = generate_and_evaluate(
        model, tokenizer, prompts, references, device,
    )
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
