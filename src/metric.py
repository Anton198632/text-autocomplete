import numpy as np
from rouge_score import rouge_scorer


def compute_rouge(predictions, references, tokenizer):
    """
    Вычисление метрик ROUGE
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,
    )

    # Декодируем предсказания и референсы
    pred_texts = [
        tokenizer.decode(pred, skip_special_tokens=True)
        for pred in predictions
    ]
    ref_texts = [
        tokenizer.decode(ref, skip_special_tokens=True)
        for ref in references
    ]

    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(pred_texts, ref_texts):
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

    # Средние значения
    avg_scores = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }

    return avg_scores
