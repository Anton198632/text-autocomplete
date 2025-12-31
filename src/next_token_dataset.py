import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def split_tokens(tokens):
    """
    Разделяет токены на input и target sequences

    Пример: [101, 3423, 434, 334, 3456, 102] ->
        input_tokens = [3423, 434, 334]
        target_tokens = [434, 334, 3456]
    """
    if len(tokens) < 3:
        raise ValueError("Слишком мало токенов для разделения")

    # Берем все токены кроме первого и последнего
    middle_tokens = tokens[1:-1]

    if len(middle_tokens) < 2:
        middle_tokens = [*middle_tokens, *middle_tokens]
        # raise ValueError("Недостаточно средних токенов")

    # Для input: все средние кроме последнего
    input_tokens = middle_tokens[:-1]

    # Для target: все средние кроме первого
    target_tokens = middle_tokens[1:]

    return input_tokens, target_tokens


class TweetDataset(Dataset):
    def __init__(self, input_ids, max_len=512, split_ratio=0.75):
        """
        Args:
            input_ids: токенизированные тексты
            max_len: максимальная длина последовательности
            split_ratio: доля текста для входа (остальное - цель)
        """
        self.input_ids = input_ids
        self.max_len = max_len
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        tokens = self.input_ids[idx]

        input_tokens, target_tokens = split_tokens(tokens)

        return {
            'input_tokens': torch.tensor(input_tokens, dtype=torch.long),
            'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
        }


def collate_fn(batch):
    inputs = [item['input_tokens'] for item in batch]
    labels = [item['target_tokens'] for item in batch]

    # Паддинг для входов
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    input_masks = (padded_inputs != 0).long()

    # Паддинг для целей
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    # -100 игнорируется при вычислении потерь в большинстве loss функций

    return {
        'input_ids': padded_inputs,
        'attention_mask': input_masks,
        'labels': padded_labels,

    }


def create_dataloader(texts, tokenizer, split_ratio=0.75):
    tokenized_texts = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,  # Добавляем [CLS] и [SEP]

    )

    dataset = TweetDataset(
        tokenized_texts['input_ids'],
        max_len=256,
        split_ratio=split_ratio,
    )

    batch_size = 256

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    print(f'Количество батчей в dataloader: {len(dataloader)}')

    # Проверка батча
    print('\nПример батча:')
    for batch in dataloader:
        print('input_ids shape:', batch['input_ids'].shape)
        print('attention_mask shape:', batch['attention_mask'].shape)
        print('labels shape:', batch['labels'].shape)

        # Пример первых двух последовательностей
        print('\nПример данных (первые 2 элемента батча):')
        for i in range(2):
            print(f"\nЭлемент {i}: ")
            input_text = tokenizer.decode(
                batch['input_ids'][i], skip_special_tokens=True,
            )
            label_text = tokenizer.decode(
                batch['labels'][i][batch['labels'][i] != -100],
                skip_special_tokens=True,
            )
            print(f"Вход: {input_text}")
            print(f"Цель: {label_text}")
        break

    return dataloader


if __name__ == '__main__':
    data = pd.read_csv('../data/val.csv')
    texts = data['tweet'] if 'tweet' in data.columns else data.iloc[:, 0]

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = create_dataloader(texts, tokenizer)
