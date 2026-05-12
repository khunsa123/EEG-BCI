import argparse
import os

from emg_stroke_recovery_monitoring.data import (
    explore_directory,
    get_raw_segmented_data,
    process_all_data_enhanced,
)
from emg_stroke_recovery_monitoring.models import train_cnn, train_mlp, train_svm
from emg_stroke_recovery_monitoring.utils import plot_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EMG stroke recovery monitoring analysis.')
    parser.add_argument(
        '--dataset-root',
        default=os.path.join(os.getcwd(), 'EMG_Reaching_Healthy_Stroke'),
        help='Path to the EMG dataset root folder containing Health_reaching and Stroke_reaching.',
    )
    parser.add_argument(
        '--output',
        default='model_comparison.png',
        help='Output filename for the consolidated comparison chart.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    healthy_path = os.path.join(dataset_root, 'Health_reaching')
    stroke_path = os.path.join(dataset_root, 'Stroke_reaching')

    healthy_subjects = explore_directory(healthy_path, 'Healthy')
    stroke_subjects = explore_directory(stroke_path, 'Stroke')

    if not healthy_subjects or not stroke_subjects:
        raise SystemExit('No subjects found in the dataset paths. Verify the dataset structure and the --dataset-root value.')

    X_full_enhanced, y_full_enhanced = process_all_data_enhanced(
        healthy_path, stroke_path, healthy_subjects, stroke_subjects
    )

    if X_full_enhanced.size == 0:
        raise SystemExit('No enhanced features were extracted. Check the dataset files and CSV parsing settings.')

    print('\nTraining Enhanced SVM...')
    svm_results = train_svm(X_full_enhanced, y_full_enhanced)
    print(f"Enhanced SVM Accuracy: {svm_results['accuracy'] * 100:.2f}%")

    print('\nTraining MLP...')
    mlp_results = train_mlp(X_full_enhanced, y_full_enhanced)
    print(f"MLP Accuracy: {mlp_results['accuracy'] * 100:.2f}%")

    print('\nExtracting raw segmented EMG windows...')
    X_raw, y_raw = get_raw_segmented_data(
        healthy_path, stroke_path, healthy_subjects, stroke_subjects
    )
    if X_raw.size == 0:
        raise SystemExit('No raw EMG segments were extracted. Check the dataset files and CSV parsing settings.')

    print('\nTraining 1D-CNN...')
    cnn_results = train_cnn(X_raw, y_raw)
    print(f"1D-CNN Accuracy: {cnn_results['accuracy'] * 100:.2f}%")

    comparison_results = {
        'Enhanced SVM': svm_results,
        'MLP': mlp_results,
        '1D-CNN': cnn_results,
    }
    plot_model_comparison(comparison_results, output_file=args.output)
    print(f'Comparison plot saved to {args.output}')


if __name__ == '__main__':
    main()
