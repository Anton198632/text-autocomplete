import evaluate
import torch
from tqdm import tqdm

from src.lstm_model import SimpleLSTM


def generate_and_evaluate(
    model, tokenizer, prompts, references, device, max_length=20, silent=False,
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
        for prompt in prompts:
            # Токенизируем с учетом attention_mask
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(device)

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

            result_texts.append(f"'{prompt}' -> '{prompt} {generated_text}'")

            all_predictions.append(generated_text)

    if not silent:
        for result in result_texts:
            print(result)

    # Вычисляем ROUGE между сгенерированными текстами и референсами
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=all_predictions, references=references)
    return scores


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
