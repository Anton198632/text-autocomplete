import evaluate
import torch
from tqdm import tqdm

from src.lstm_model import SimpleLSTM
from src.metric import compute_rouge


def generate_and_evaluate(
    model, tokenizer, prompts, references, device, silent=False,
):
    """
    Генерация текста и вычисление ROUGE
    """
    model.eval()
    all_predictions = []

    result_texts = []

    with torch.no_grad():
        prompts = (
            prompts if silent else tqdm(prompts, desc='Генерация текста')
        )

        for i, prompt in enumerate(prompts):
            # Токенизируем с учетом attention_mask
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(device)

            ref_inputs = tokenizer(
                references[i],
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(device)

            max_length = ref_inputs['input_ids'].shape[1]

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Генерируем продолжение
            if type(model) == SimpleLSTM:
                generated_ids = model.generate(
                    input_ids, num_tokens=max_length,
                )
            else:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    num_beams=1,
                    no_repeat_ngram_size=2,
                    # early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,  # Включаем случайную выборку
                    temperature=0.6,  # Более консервативная
                    # Контроль "креативности": 0.7-0.9 для баланса
                    top_k=30,  # Ограничиваем выбор сверху
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=1.3,  # Штраф за повторения
                )

            # Декодируем сгенерированный текст (только продолжение)
            generated_text = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            result_texts.append(
                f"'{prompt}' -> '{prompt} {generated_text}'"
            )

            all_predictions.append(generated_text)

    if not silent:
        for result in result_texts:
            print(result)

    # Вычисляем ROUGE между сгенерированными текстами и референсами
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=all_predictions, references=references)
    return scores


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
