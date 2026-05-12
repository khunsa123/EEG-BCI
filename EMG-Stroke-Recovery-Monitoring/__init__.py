"""EMG stroke recovery monitoring package."""

from .data import (
    explore_directory,
    process_all_data_enhanced,
    get_raw_segmented_data,
    load_target_dataframe,
)
from .features import extract_time_domain_features, extract_frequency_features
from .models import EMGClassifier, EMG_1DCNN, train_svm, train_mlp, train_cnn
from .utils import plot_model_comparison
