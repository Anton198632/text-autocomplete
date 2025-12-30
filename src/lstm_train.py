import torch
from rouge_score import rouge_scorer
from torch import optim, nn, no_grad, argmax
from tqdm import tqdm

from src.eval_lstm import evaluate_with_rouge, generate_text
from src.metric import compute_rouge
from src.train_history import TrainHistory


def train_model_with_rouge(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    device,
    num_epochs=10,
    learning_rate=0.001,
    eval_every=100,
    save_dir="checkpoints",
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
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Режим обучения
        model.train()
        epoch_train_loss = 0
        total_train_tokens = 0

        # Списки для накопления предсказаний и таргетов для ROUGE
        train_predictions = []
        train_references = []

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

            # Получаем размер labels
            batch_size_labels, seq_len_labels = labels.shape

            # ПРОБЛЕМА: seq_len_logits и seq_len_labels могут отличаться!
            # Решение: используем min(seq_len_logits, seq_len_labels)
            seq_len = min(seq_len_logits, seq_len_labels)

            # Обрезаем logits и labels до одинаковой длины
            logits_trimmed = logits[:, :seq_len, :]
            labels_trimmed = labels[:, :seq_len]

            # Создаем маску для валидных позиций (где labels != -100)
            labels_mask = labels_trimmed != -100

            # НОВОЕ: Проверяем, есть ли валидные позиции
            if not labels_mask.any():
                # Пропускаем батч без валидных таргетов
                train_pbar.set_postfix({
                    'loss': 'skip',
                    'avg_loss': (f'{epoch_train_loss / (batch_idx + 1) / batch_size:.4f}')
                })
                continue

            # Подготавливаем логиты и таргеты для loss
            # Сначала flatten логитов
            logits_flat = logits_trimmed.reshape(-1, vocab_size)
            labels_flat = labels_trimmed.reshape(-1)

            # Создаем маску для игнорирования padding в loss
            loss_mask = labels_flat != -100

            # Берем только валидные элементы
            valid_logits = logits_flat[loss_mask]
            valid_labels = labels_flat[loss_mask]

            # Проверяем, что есть валидные данные
            if valid_logits.size(0) == 0:
                train_pbar.set_postfix({'loss': 'skip',
                                        'avg_loss': f'{epoch_train_loss / (batch_idx + 1) / batch_size:.4f}'})
                continue

            # Вычисляем потери
            loss = criterion(valid_logits, valid_labels)

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
            # if batch_idx % 10 == 0:
            #     with no_grad():
            #         # Предсказываем следующий токен
            #         predicted_ids = argmax(logits, dim=-1)
            #
            #         # Сохраняем только нетривиальные примеры
            #         for i in range(
            #                 min(2, batch_size)):  # Берем 2 примера из батча
            #             # Берем последний предсказанный токен
            #             pred_seq = predicted_ids[i]
            #             ref_seq = labels[i]
            #
            #             # Фильтруем padding токены
            #             mask = ref_seq != -100
            #             if mask.sum() > 0:
            #                 train_predictions.append(pred_seq[mask])
            #                 train_references.append(ref_seq[mask])

            # Обновляем прогресс-бар
            avg_batch_loss = epoch_train_loss / ((batch_idx + 1) * batch_size)
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{avg_batch_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Промежуточная валидация и вычисление ROUGE
            if global_step % eval_every == 0:
                # Вычисляем ROUGE на тренировочной выборке
                if len(train_predictions) > 0:
                    train_rouge = compute_rouge(
                        train_predictions, train_references, tokenizer,
                    )

                    # Валидация
                    val_loss, val_rouge = evaluate_with_rouge(
                        model, val_dataloader, tokenizer, device, criterion
                    )

                    print(f"\nStep {global_step}:")
                    print(f"  Train Loss: {avg_batch_loss:.4f}")
                    print(f"  Train ROUGE: R1={train_rouge['rouge1']:.3f}, "
                          f"R2={train_rouge['rouge2']:.3f}, "
                          f"RL={train_rouge['rougeL']:.3f}")
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"  Val ROUGE: R1={val_rouge['rouge1']:.3f}, "
                          f"R2={val_rouge['rouge2']:.3f}, "
                          f"RL={val_rouge['rougeL']:.3f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

                    # Сохраняем историю
                    history.train_loss.append(avg_batch_loss)
                    history.val_loss.append(val_loss)
                    history.train_rouge.append(train_rouge)
                    history.val_rouge.append(val_rouge)
                    history.learning_rate.append(
                        optimizer.param_groups[0]['lr']
                    )

                    # Сохраняем лучшую модель
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_rouge': val_rouge,
                        }, f"{save_dir}/best_model.pt")
                        print(
                            f"  ✅ Сохранена лучшая модель"
                            f" (val_loss={val_loss:.4f})"
                        )

                    # Генерация примеров
                    print("\n  Примеры генерации:")
                    example_texts = [
                        "I love", "The weather is", "Machine learning",
                    ]
                    for text in example_texts:
                        generated = generate_text(
                            model, tokenizer, text, device,
                        )
                        print(f"    '{text}' -> '{generated}'")
                    print()

        # Конец эпохи - полная валидация
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} завершена")

        # Полная валидация
        val_loss, val_rouge = evaluate_with_rouge(
            model, val_dataloader, tokenizer, device, criterion
        )

        avg_train_loss = epoch_train_loss / len(train_dataloader.dataset)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val ROUGE-1: {val_rouge['rouge1']:.3f}")
        print(f"Val ROUGE-2: {val_rouge['rouge2']:.3f}")
        print(f"Val ROUGE-L: {val_rouge['rougeL']:.3f}")
        print(f"{'=' * 60}\n")

        # Сохраняем чекпоинт эпохи
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_rouge': val_rouge,
            'history': history.to_dict(),
        }, f"{save_dir}/epoch_{epoch + 1}.pt")

    return model, history


def train_model_with_rouge_v2(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    device,
    num_epochs=10,
    learning_rate=0.001,
    eval_every=100,
    save_dir="checkpoints",
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
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Режим обучения
        model.train()
        epoch_train_loss = 0
        total_train_tokens = 0

        # Списки для накопления предсказаний и таргетов для ROUGE
        train_predictions = []
        train_references = []

        # Прогресс-бар для эпохи
        train_pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
        )

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,
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
            # Вычисление loss
            # Переформатируем logits и labels для CrossEntropyLoss
            batch_size, seq_len, vocab_size = logits.shape

            loss = criterion(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )

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
            if batch_idx % 1 == 0:
                with no_grad():
                    # Предсказываем следующий токен
                    predicted_ids = argmax(logits, dim=-1)

                    # Сохраняем только нетривиальные примеры
                    for i in range(
                            min(2, batch_size)):  # Берем 2 примера из батча
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
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Промежуточная валидация и вычисление ROUGE
            if global_step % eval_every == 0:
                # Вычисляем ROUGE на тренировочной выборке
                if len(train_predictions) > 0:
                    train_rouge = compute_rouge(
                        train_predictions, train_references, tokenizer,
                    )

                    # Валидация
                    val_loss, val_rouge = evaluate_with_rouge(
                        model, val_dataloader, tokenizer, device, criterion
                    )

                    print(f"\nStep {global_step}:")
                    print(f"  Train Loss: {avg_batch_loss:.4f}")
                    print(f"  Train ROUGE: R1={train_rouge['rouge1']:.3f}, "
                          f"R2={train_rouge['rouge2']:.3f}, "
                          f"RL={train_rouge['rougeL']:.3f}")
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"  Val ROUGE: R1={val_rouge['rouge1']:.3f}, "
                          f"R2={val_rouge['rouge2']:.3f}, "
                          f"RL={val_rouge['rougeL']:.3f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

                    # Сохраняем историю
                    history.train_loss.append(avg_batch_loss)
                    history.val_loss.append(val_loss)
                    history.train_rouge.append(train_rouge)
                    history.val_rouge.append(val_rouge)
                    history.learning_rate.append(
                        optimizer.param_groups[0]['lr']
                    )

                    # Сохраняем лучшую модель
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_rouge': val_rouge,
                        }, f"{save_dir}/best_model.pt")
                        print(
                            f"  ✅ Сохранена лучшая модель"
                            f" (val_loss={val_loss:.4f})"
                        )

                    # Генерация примеров
                    print("\n  Примеры генерации:")
                    example_texts = [
                        "I love", "The weather is", "Machine learning",
                    ]
                    for text in example_texts:
                        generated = generate_text(
                            model, tokenizer, text, device,
                        )
                        print(f"    '{text}' -> '{generated}'")
                    print()

        # Конец эпохи - полная валидация
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} завершена")

        # Полная валидация
        val_loss, val_rouge = evaluate_with_rouge(
            model, val_dataloader, tokenizer, device, criterion
        )

        avg_train_loss = epoch_train_loss / len(train_dataloader.dataset)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val ROUGE-1: {val_rouge['rouge1']:.3f}")
        print(f"Val ROUGE-2: {val_rouge['rouge2']:.3f}")
        print(f"Val ROUGE-L: {val_rouge['rougeL']:.3f}")
        print(f"{'=' * 60}\n")

        # Сохраняем чекпоинт эпохи
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_rouge': val_rouge,
            'history': history.to_dict(),
        }, f"{save_dir}/epoch_{epoch + 1}.pt")

    return model, history