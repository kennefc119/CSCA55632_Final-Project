import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define feature key names
def get_feature_key_names():
    types = ["close", "open", "high", "low", "volume"]
    feature_key_names = []
    for category in types:
        feature_key_names.append(f'{category}_relative_max')
        feature_key_names.append(f'{category}_relative_min')
        feature_key_names.append(f'{category}_change')
        feature_key_names.append(f'pairwise_{category}_mean')
    feature_key_names.append('candle_body_ratio')
    feature_key_names.append('candle_pairwise_body_change_mean')
    return feature_key_names

# Group features by their key names
def get_grouped_features(df, feature_key_names):
    grouped_features = []
    for i,key in enumerate(feature_key_names):
        sample_features = [col for col in df.columns if key in col]
        grouped_features.append(sample_features)
    
    # flatten the list of lists
    grouped_features = [item for sublist in grouped_features for item in sublist]
    
    return grouped_features

# log-transform features and clip outliers
def log_transform_and_clip(df, columns):
    for col in columns:
        df[col] = np.log1p(df[col])
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower, upper)
    return df

# Stack feature values for visualization
def get_stack_feature_values(df, grouped_features, feature_key_names):
    stack_feature_values = []
    for sample_features,key in zip(grouped_features,feature_key_names):
        # Get the features for the specific key (stacked across all sliding windows)
        sample_features = [col for col in df.columns if key in col]
        stacked_values = df[sample_features].values.flatten()
        stacked_values = stacked_values[~pd.isnull(stacked_values)]
        stack_feature_values.append(stacked_values)

    return stack_feature_values

# Plot histograms for all features
def plot_histograms(stack_feature_values, feature_key_names):
    n_cols = 2
    n_rows = (len(stack_feature_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
    axes = axes.flatten()

    for i, values in enumerate(stack_feature_values):
        sns.histplot(values, kde=True, bins=50, ax=axes[i])
        axes[i].set_title(f"Histogram: {feature_key_names[i]}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# log-transform features and clip outliers
def log_transform_and_clip(df, columns, epsilon=1e-8):
    for col in columns:
        df[col] = np.log1p(df[col] + epsilon)
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower, upper)
    return df
