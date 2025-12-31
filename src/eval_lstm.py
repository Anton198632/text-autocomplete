import torch
from tqdm import tqdm

from src.metric import compute_rouge


def evaluate_with_rouge(model, dataloader, tokenizer, device, criterion):
    """
    Оценка модели с вычислением ROUGE
    """
    model.eval()
    total_loss = 0
    total_batches = 0

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids)
            batch_size, seq_len_logits, vocab_size = logits.shape

            # Вычисляем потери
            loss = criterion(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )

            total_loss += loss.item() * batch_size
            total_batches += batch_size

            # Предсказания для ROUGE
            predicted_ids = torch.argmax(logits, dim=-1)

            # Сохраняем предсказания и референсы
            for i in range(batch_size):
                # Фильтруем padding токены в референсах
                mask = labels[i] != -100
                if mask.sum() > 0:
                    all_predictions.append(predicted_ids[i][mask])
                    all_references.append(labels[i][mask])

    # Средний loss
    avg_loss = total_loss / len(dataloader.dataset)

    # Вычисляем ROUGE
    if len(all_predictions) > 0:
        rouge_scores = compute_rouge(
            all_predictions, all_references, tokenizer,
        )
    else:
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    return avg_loss, rouge_scores


def generate_text(model, tokenizer, prompt, device, max_length=20):
    """
    Генерация текста по промпту
    """
    model.eval()

    # Токенизация промпта
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        # Генерируем продолжение
        generated_ids = model.generate(
            input_ids,
            num_tokens=max_length,
        )

    # Декодируем результат
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True,
    )

    return generated_text
