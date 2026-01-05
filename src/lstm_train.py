import torch
from torch import nn, optim, no_grad, argmax
from tqdm import tqdm

from src.data_utils import split_text_3_4
from src.eval_lstm import generate_and_evaluate, generate_text, \
    evaluate_with_rouge
from src.metric import compute_rouge
from src.train_history import TrainHistory


def train_model_with_rouge(
    model,
    train_dataloader,
    val_dataloader,
    selected_texts,
    tokenizer,
    device,
    num_epochs=10,
    learning_rate=0.001,
    eval_every=100,
    save_dir='checkpoints',
):
    """
    Обучение модели с вычислением ROUGE метрик
    """
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # История обучения
    history = TrainHistory()

    print(f"Начинаем обучение на {device}...")
    print(f"Размер словаря: {model.vocab_size}")

    global_step = 0

    # Промты и референсы для промежуточных оценок ROUGE
    prompts = []
    references = []
    for text in selected_texts:
        split_data = split_text_3_4(text, tokenizer)
        prompts.append(split_data['prompt'])
        references.append(split_data['target'])

    # Списки для накопления предсказаний и таргетов для ROUGE
    train_predictions = []
    train_references = []

    for epoch in range(num_epochs):
        # Режим обучения
        model.train()

        epoch_train_loss = 0
        total_train_tokens = 0

        # Прогресс-бар для эпохи
        train_pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
        )

        for batch_idx, batch in enumerate(train_pbar):
            global_step += 1

            # Перенос данных
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)
            batch_size, seq_len_logits, vocab_size = logits.shape

            # Вычисляем потери
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping для стабильности
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Оптимизация
            optimizer.step()

            # Собираем статистику
            batch_loss = loss.item()
            batch_tokens = attention_mask.sum().item()

            epoch_train_loss += batch_loss * batch_size
            total_train_tokens += batch_tokens

            # Сохраняем предсказания для ROUGE (только для последнего токена)
            # Сохраняем каждые 10 батчей для экономии памяти
            if batch_idx % 10 == 0:
                with no_grad():
                    # Предсказываем следующий токен
                    predicted_ids = argmax(logits, dim=-1)

                    # Берем 2 примера из батча
                    for i in range(min(2, batch_size)):
                        # Берем последний предсказанный токен
                        pred_seq = predicted_ids[i]
                        ref_seq = labels[i]

                        # Фильтруем padding токены
                        mask = ref_seq != -100
                        if mask.sum() > 0:
                            train_predictions.append(pred_seq[mask])
                            train_references.append(ref_seq[mask])

            # Обновляем прогресс-бар
            avg_batch_loss = epoch_train_loss / ((batch_idx + 1) * batch_size)
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{avg_batch_loss:.4f}',
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            })

            # Промежуточное вычисление ROUGE
            if global_step % eval_every == 0:
                # Вычисляем ROUGE на тренировочной выборке
                if len(train_predictions) > 0:
                    train_rouge = compute_rouge(
                        train_predictions, train_references, tokenizer,
                    )

                    # Сохраняем историю
                    history.train_loss.append(avg_batch_loss)
                    history.train_rouge.append(train_rouge)
                    history.learning_rate.append(
                        optimizer.param_groups[0]['lr']
                    )

        # Конец эпохи - полная валидация
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} завершена")

        # Полная валидация
        val_loss, val_rouge = evaluate_with_rouge(
            model, val_dataloader, tokenizer, device, criterion
        )

        avg_train_loss = epoch_train_loss / len(train_dataloader.dataset)

        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val ROUGE-1: {val_rouge['rouge1']:.3f}")
        print(f"Val ROUGE-2: {val_rouge['rouge2']:.3f}")
        print(f"Val ROUGE-L: {val_rouge['rougeL']:.3f}")
        print(f"{'=' * 60}\n")

        # Генерация примеров
        print('\n  Примеры генерации:')
        example_texts = [
            'I love', 'The weather is', 'Machine learning',
        ]
        for text in example_texts:
            generated = generate_text(
                model, tokenizer, text, device,
            )
            print(f"    '{text}' -> '{generated}'")
        print()

        # Сохраняем чекпоинт эпохи
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history.to_dict(),
        }, f"{save_dir}/epoch_{epoch + 1}.pt")

    return model, history
