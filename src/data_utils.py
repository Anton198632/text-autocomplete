import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_text(work_dir):
    cleaned_tweets = []

    with open(f'{work_dir}tweets.txt', 'r') as f:
        tweets = f.readlines()

        print('Обработка текста...')

        for tweet in tqdm(tweets):
            # Оставляем только буквы и пробелы,
            # заменяем всё остальное на пробел
            text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', str(tweet))
            # Заменяем множественные пробелы на один
            text = re.sub(r'\s+', ' ', text)
            # Приводим к нижнему регистру и убираем лишние пробелы
            cleaned_tweets.append(text.strip().lower())

    df = pd.DataFrame({'tweet': cleaned_tweets})

    df.to_csv(
        f'{work_dir}dataset_processed.csv', index=False, encoding='utf-8',
    )

    # Случайное перемешивание данных
    print('\nПеремешиваем данные...')
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Сначала отделяем 80% для тренировочного набора
    train_df, temp_df = train_test_split(
        df_shuffled,
        test_size=0.2,  # 20% для val + test
        random_state=42,
        shuffle=True,
    )

    # Затем оставшиеся 20% делим пополам на validation и test (10% + 10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # половина от 20% = 10%
        random_state=42,
        shuffle=True,
    )

    # Проверка размеров наборов
    print(
        f'Размер тренировочного набора: '
        f'{len(train_df)} ({len(train_df) / len(df) * 100: .1f}%)',
    )
    print(
        f'Размер валидационного набора: '
        f'{len(val_df)} ({len(val_df) / len(df) * 100: .1f}%)',
    )
    print(
        f'Размер тестового набора: '
        f'{len(test_df)} ({len(test_df) / len(df) * 100: .1f}%)',
    )

    print('\nСохраняем разделенные датасеты...')

    # Тренировочный набор
    train_file = f'{work_dir}train.csv'
    train_df.to_csv(train_file, index=False, encoding='utf-8')
    print(f'Тренировочный набор сохранен в: {train_file}')

    # Валидационный набор
    val_file = f'{work_dir}val.csv'
    val_df.to_csv(val_file, index=False, encoding='utf-8')
    print(f'Валидационный набор сохранен в: {val_file}')

    # Тестовый набор
    test_file = f'{work_dir}test.csv'
    test_df.to_csv(test_file, index=False, encoding='utf-8')
    print(f'Тестовый набор сохранен в: {test_file}')


def split_text_3_4(text, tokenizer):
    # Разбиваем текст на слова
    words = text.split()
    split_point = len(words) * 3 // 4

    # Формируем промпт и таргет
    prompt = ' '.join(words[:split_point])
    target = ' '.join(words[split_point:])

    # Токенизируем и считаем токены
    prompt_tokens = tokenizer(
        prompt, return_tensors='pt', add_special_tokens=False,
    )['input_ids'][0]
    target_tokens = tokenizer(
        target, return_tensors='pt', add_special_tokens=False,
    )['input_ids'][0]

    prompt_token_count = len(prompt_tokens)
    target_token_count = len(target_tokens)
    total_token_count = prompt_token_count + target_token_count

    return {
        'prompt': prompt,
        'target': target,
        'prompt_tokens': prompt_token_count,
        'target_tokens': target_token_count,
        'total_tokens': total_token_count,
    }


if __name__ == '__main__':
    process_text('../data/')
