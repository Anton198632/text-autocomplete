import os

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer

from src.eval_lstm import evaluate_with_rouge, generate_text
from src.lstm_model import SimpleLSTM
from src.lstm_train import train_model_with_rouge, train_model_with_rouge_v2
from src.next_token_dataset import create_dataloader


if __name__ == '__main__':
    # Создаем директорию для чекпоинтов
    save_directory = "checkpoints"
    os.makedirs(save_directory, exist_ok=True)

    print('Загрузка тренировочного датасета...')
    train_data = pd.read_csv('../data/train.csv')
    train_texts = (
        train_data['tweet'] if 'tweet' in train_data.columns
        else train_data.iloc[:, 0]
    )

    print('Загрузка валидационного датасета...')
    val_data = pd.read_csv('../data/val.csv')
    val_texts = (
        val_data['tweet'] if 'tweet' in val_data.columns
        else val_data.iloc[:, 0]
    )

    # Загружаем токенизатор
    print('Загрузка токенизатора...')
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Добавляем специальные токены для начала/конца
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
        num_layers=2
    )

    print(f"\nМодель создана:")
    print(f"  Параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Размер словаря: {vocab_size}")

    # Обучаем модель
    trained_model, history = train_model_with_rouge_v2(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
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
    print("\n" + "=" * 60)
    print("Финальная оценка модели:")
    print("=" * 60)

    final_val_loss, final_val_rouge = evaluate_with_rouge(
        trained_model,
        val_dataloader,
        tokenizer,
        device,
        criterion=nn.CrossEntropyLoss(ignore_index=-100)
    )

    print(f"Val Loss: {final_val_loss:.4f}")
    print(f"Val ROUGE-1: {final_val_rouge['rouge1']:.3f}")
    print(f"Val ROUGE-2: {final_val_rouge['rouge2']:.3f}")
    print(f"Val ROUGE-L: {final_val_rouge['rougeL']:.3f}")

    # Примеры генерации
    print("\nПримеры генерации текста:")
    test_prompts = [
        "The future of AI",
        "I believe that",
        "In my opinion",
        "The best thing about"
    ]

    for prompt in test_prompts:
        generated = generate_text(trained_model, tokenizer, prompt, device)
        print(f"  '{prompt}' -> '{generated}'")




