import torch
from torch import nn, softmax


class SimpleLSTM(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2,
    ):
        """
        Модель LSTM для предсказания следующего токена

        Args:
            vocab_size: размер словаря
            embedding_dim: размерность эмбеддингов
            hidden_dim: размерность скрытого состояния LSTM
            num_layers: количество слоев LSTM
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Эмбеддинг токенов
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Полносвязный слой для предсказания следующего токена
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        Forward pass для предсказания следующего токена

        Args:
            input_ids: тензор токенов [batch_size, seq_len]

        Returns:
            logits: логиты для всех позиций [batch_size, seq_len, vocab_size]
        """
        # Эмбеддинг токенов
        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedding(input_ids)

        # Пропускаем через LSTM
        # [batch_size, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(embeddings)

        # Предсказание следующего токена для каждой позиции
        # [batch_size, seq_len, vocab_size]
        logits = self.fc(lstm_out)

        return logits

    def predict_next(self, input_ids):
        """
        Предсказание следующего токена для заданной последовательности

        Args:
            input_ids: тензор токенов [batch_size, seq_len]

        Returns:
            next_token: следующий токен [batch_size, 1]
            probabilities: вероятности всех токенов [batch_size, vocab_size]
        """
        # Получаем логиты для последней позиции
        logits = self.forward(input_ids)
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Применяем softmax для получения вероятностей
        probs = softmax(last_logits, dim=-1)

        # Берем токен с максимальной вероятностью
        # [batch_size, 1]
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        return next_token, probs

    def generate(self, start_ids, num_tokens=10):
        """
        Генерация нескольких токенов

        Args:
            start_ids: начальная последовательность токенов
            num_tokens: сколько токенов сгенерировать

        Returns:
            all_ids: исходная + сгенерированная последовательность
        """
        # Режим оценки
        self.eval()

        # Если start_ids - это список, преобразуем в тензор
        if not isinstance(start_ids, torch.Tensor):
            start_ids = torch.tensor(start_ids)

        # Добавляем batch dimension если нужно
        if len(start_ids.shape) == 1:
            start_ids = start_ids.unsqueeze(0)  # [1, seq_len]

        # Копируем начальные токены
        current_ids = start_ids.clone()
        all_ids = start_ids.clone()

        with torch.no_grad():  # Отключаем вычисление градиентов
            for _ in range(num_tokens):
                # Предсказываем следующий токен
                next_token, _ = self.predict_next(current_ids)

                # Добавляем новый токен
                all_ids = torch.cat([all_ids, next_token], dim=1)

                # Обновляем текущую последовательность
                # (добавляем только что сгенерированный токен)
                current_ids = torch.cat(
                    [current_ids[:, 1:], next_token], dim=1,
                )

        return all_ids
