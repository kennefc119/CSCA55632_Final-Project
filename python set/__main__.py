import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_func import fetch_clean_data, sort_by_date, get_us_equity_tickers, clean_lid_symbol,compute_sliding_window_features,clean_ticker_matrix ,ticker_matrix_creation
from EDA_func import get_feature_key_names, get_grouped_features, log_transform_and_clip, get_stack_feature_values, plot_histograms
from model_func import pca_transform,simple_Knearest, knn_hyper_tune,knn_hyper_tune_single,kmeans_elbow_method,Kmeans_clustering,Agglomerative_clustering,Agglomerative_clustering_hyper_tune

# Fetch and clean data###################################################################################################
df = fetch_clean_data("AAPL", period = '120d', interval="1d")
df = sort_by_date(df)
print (f'Shape of data {df.shape}')
print (df.head(5))
print ("-"*40)

# Fetch and clean tickers
tickers = get_us_equity_tickers()
tickers = [s for s in tickers if clean_lid_symbol(s)]
print (f"Total tickers fetched: {len(tickers)}")
print (f'First 10 tickers: {tickers[:10]}') 
print ("-"*40)

# Parameters for sliding window feature engineering
_partition_days = 15
_step_shift_days = 5

# Compute derived features
ticker_vector = compute_sliding_window_features(df, shift_days = _step_shift_days, partition_days = _partition_days)
print (ticker_vector.shape)
print (ticker_vector.isnull().sum().sum())
print (f'Ticker Vector Shape: {ticker_vector.shape}')
print (f'First 5 Column Names: {list(ticker_vector.columns[:5])}')
print ("-"*40)

# Create ticker subset matrix by defining analysis windows in days. To evaluate the approach###########################################
_local_run = True # True if running locally with pre-downloaded data, False to fetch from yfinance
analysis_days_split = 15
_step_shift_days = 5
total_lookback_days = 120
file_name = "data/full_ticker_matrix.csv"

# ticker_matrix_creation(tickers,analysis_days_split,_step_shift_days,_partition_days,_period = total_lookback_days,_interval= '1d',local_run = _local_run, output_file = file_name)
print (f"Full Ticker matrix creation completed file saved: {file_name}")

#EDA and Visualization########################################################################################################
# Retrieve and clean the engineered feature matrice
full_ticker_matrix = pd.read_csv("data/full_ticker_matrix.csv")
full_ticker_matrix = clean_ticker_matrix(full_ticker_matrix)
print (f'full_ticker_matrix.shape: {full_ticker_matrix.shape}')

# Stack feature values across sliding windows and plot histograms
full_df = full_ticker_matrix.copy()
feature_key_names = get_feature_key_names()
grouped_features = get_grouped_features(full_df, feature_key_names)
stack_feature_values = get_stack_feature_values(full_df, grouped_features, feature_key_names)
plot_histograms(stack_feature_values, feature_key_names)

# Log-transform and clip outliers
full_df = log_transform_and_clip(full_df, grouped_features)
stack_feature_values = get_stack_feature_values(full_df, grouped_features, feature_key_names)
plot_histograms(stack_feature_values, feature_key_names)

# Correlation heatmaps for different feature categories
categories = ["close", "open", "high", "low", "vol"]
for category in categories:
    features = [col for col in full_df.columns if category in col]
    corr_matrix = full_df[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.title(f"Correlation Heatmap for {category.capitalize()} Features")
    plt.show()
    
#Modelling########################################################################################################
#Prepare data for KNN
X = full_df.set_index("Ticker")

# KNN Model
pca_components = 50
X_pca = pca_transform(X, pca_components)
model, KN_result = simple_Knearest(X, X_pca, "TSLA")

# Hyperparameter tuning for KNN
knn_hyper_tune(X,X_pca, ["TSLA","HOOD","AMD"])
nr_most_similiar = 6
knn_hyper_tune_single(X,["TSLA"],X_pca,nr_most_similiar)

# K-Means Clustering
kmeans_elbow_method(X_pca)
k = 278
nr_most_similiar = 6
pca_components = 20
Kmeans_clustering(X, k, pca_components, ["TSLA","HOOD","AMD"], nr_most_similiar)

#Algorithm functions
nr_most_similiar = 6
pca_components = 30
X_pca = pca_transform(X, pca_components)
Agglomerative_clustering(X_pca, ["TSLA","HOOD","AMD"], nr_most_similiar)

#Algorithm clustering Hyperparameter tuning
nr_most_similiar = 1
pca_components = [20,50,100,150]
Agglomerative_clustering_hyper_tune(X,X_pca, ['TSLA'], nr_most_similiar,pca_components)


