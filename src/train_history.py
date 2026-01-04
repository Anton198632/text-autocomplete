import matplotlib.pyplot as plt


class TrainHistory:
    def __init__(self):
        self.train_loss = []
        self.train_rouge = []
        self.learning_rate = []

    def to_dict(self):
        return {
            'train_loss': self.train_loss,
            'train_rouge': self.train_rouge,
            'learning_rate': self.learning_rate,
        }

    def plot_training_history(self):
        """
        Визуализация истории обучения
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.train_loss, label='Train Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # ROUGE-1
        train_r1 = [h['rouge1'] for h in self.train_rouge]
        axes[0, 1].plot(train_r1, label='Train ROUGE-1')
        axes[0, 1].set_title('ROUGE-1')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # ROUGE-2
        train_r2 = [h['rouge2'] for h in self.train_rouge]
        axes[1, 0].plot(train_r2, label='Train ROUGE-2')
        axes[1, 0].set_title('ROUGE-2')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # ROUGE-L
        train_rl = [h['rougeL'] for h in self.train_rouge]
        axes[1, 1].plot(train_rl, label='Train ROUGE-L')
        axes[1, 1].set_title('ROUGE-L')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
