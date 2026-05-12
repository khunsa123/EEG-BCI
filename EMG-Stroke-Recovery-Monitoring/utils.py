import os
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(results: Dict[str, Dict], output_file: str = 'model_comparison.png') -> None:
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in model_names]
    f1_scores = [results[name]['f1_score'] * 100 for name in model_names]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    x = range(len(model_names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], accuracies, width, label='Accuracy', color='#3498db')
    ax.bar([i + width / 2 for i in x], f1_scores, width, label='F1 Score', color='#e74c3c')
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Percentage')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for idx, name in enumerate(model_names):
        cm = results[name]['confusion_matrix']
        ax_cm = fig.add_subplot(2, len(model_names), len(model_names) + idx + 1)
        sns.heatmap(
            cm,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=['Healthy', 'Stroke'],
            yticklabels=['Healthy', 'Stroke'],
            ax=ax_cm,
        )
        ax_cm.set_title(f'{name} Confusion Matrix')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
