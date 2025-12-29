import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TweetDataset(Dataset):
    def __init__(self, input_ids, max_len, split_ratio=0.75):
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
        sequence = self.input_ids[idx][:self.max_len]
        split_point = int(len(sequence) * self.split_ratio)

        # Входная часть
        input_part = sequence[:split_point]
        # Целевая часть
        target_part = sequence[split_point:]  # Для дополнения текста

        return {
            'input_ids': torch.tensor(input_part, dtype=torch.long),
            'labels': torch.tensor(target_part, dtype=torch.long),
        }


def collate_fn(batch):
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Паддинг для входов
    input_lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    input_masks = (padded_inputs != 0).long()

    # Паддинг для целей
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    # -100 игнорируется при вычислении потерь в большинстве loss функций

    return {
        'input_ids': padded_inputs,
        'attention_mask': input_masks,
        'labels': padded_labels,
        'lengths': input_lengths,
    }


def create_dataloader(texts, split_ratio=0.75):
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Добавляем специальные токены для начала/конца
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

    batch_size = 64

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
        print('lengths shape:', batch['lengths'].shape)

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

    return dataloader, tokenizer


if __name__ == '__main__':
    data = pd.read_csv('../data/val.csv')
    texts = data['tweet'] if 'tweet' in data.columns else data.iloc[:, 0]

    dataloader = create_dataloader(texts)
