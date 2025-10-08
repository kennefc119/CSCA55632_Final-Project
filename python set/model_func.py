import pandas as pd
import math
import matplotlib.pyplot as plt
import mplfinance as mpf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances

# PCA transformation function###########################################################################################
def pca_transform(X, pca_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(X_pca, index=X.index, columns=[f'PC{i+1}' for i in range(pca_components)])
    return X_pca

# Function to plot candlestick charts for a list of tickers
def plot_candlestick(tickers, tags,title_name,nr_cols=3):
    n = len(tickers)
    cols = nr_cols
    rows = math.ceil(n / nr_cols)
    
    fig, axes = plt.subplots(nrows=rows, ncols=nr_cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, (ticker, tag) in enumerate(zip(tickers, tags)):
        ax = axes[i]
        df = pd.read_csv(f"data/{ticker}.csv", parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        df = df[['Open', 'High', 'Low', 'Close']].tail(120)

        mpf.plot(
            df,
            type='candle',
            style='charles',
            ax=ax,
            volume=False,
            xrotation=0,
            ylabel='Price'
        )
        ax.set_title(f"{ticker} - {title_name} - {tag}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Simple K-nearest Neighbors function###########################################################################################
def simple_Knearest(X, X_pca, target_ticker):
    # Model to find similar stocks
    model = NearestNeighbors(
        n_neighbors=5,
        metric='euclidean',
        algorithm='auto',
        n_jobs=-1
    )
    model.fit(X_pca)

    # Find similar stocks using KNN
    distances, indices = model.kneighbors([X_pca.loc[target_ticker]])
    KN_result = format_result_distance(X,distances,indices)
    print (KN_result)

    return model, KN_result

# Format the KNN result into a DataFrame
def format_result_distance(X,distances,indices):
    similar_tickers = X.iloc[indices[0][1:]].index.tolist()
    similar_distances = distances[0][1:]

    KN_result = pd.DataFrame({
        "Ticker": similar_tickers,
        "Similarity": similar_distances
    })
    return KN_result

# List of distance metrics for KNN
def get_model_metrics():
    return [
        'euclidean',
        'manhattan',
        'chebyshev',
        'cosine',
        'hamming',
        'jaccard'
    ]


# K-nearest hyper-tune modelling###########################################################################################
def knn_hyper_tune(X,X_pca,target_tickers):   
    model_metrics = get_model_metrics()

    for target_ticker in target_tickers:
        closest_tickers = []
        for model_metric in model_metrics:
            model = NearestNeighbors(
                n_neighbors=5,
                metric=model_metric,
                algorithm='auto',
                n_jobs=-1
            ).fit(X_pca)

            # Find similar stocks using KNN
            distances, indices = model.kneighbors([X_pca.loc[target_ticker]])
            KN_result = format_result_distance(X,distances,indices)
            
            closest_ticker = KN_result['Ticker'][0]
            closest_tickers.append(closest_ticker)
            
        plot_candlestick ([target_ticker],['Target'],"NearestNeigh",nr_cols=3)
        plot_candlestick (closest_tickers,model_metrics,"NearestNeigh",nr_cols=3)
        
def knn_hyper_tune_single(X,target_tickers,X_pca,nr_most_similiar):
    for target_ticker in target_tickers:
        for _metric in get_model_metrics():
            model = NearestNeighbors(
                n_neighbors=nr_most_similiar+1,
                metric=_metric,
                algorithm='auto',
                n_jobs=-1
            ).fit(X_pca)

            # Find similar stocks using KNN
            distances, indices = model.kneighbors([X_pca.loc[target_ticker]])
            KN_result = format_result_distance(X,distances,indices)
            closest_tickers = KN_result['Ticker'][:nr_most_similiar+1]
                
            plot_candlestick ([target_ticker],['Target'],"NearestNeigh",nr_cols=3)
            plot_candlestick (closest_tickers,[f"{_metric} - Rank{i}" for i in range(1, nr_most_similiar+1)],"NearestNeigh",nr_cols=3)
            

#KMeans clustering function###########################################################################################
def kmeans_elbow_method(X_pca, k_range = range(250,300)):
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_pca)
        inertia.append(kmeans.inertia_)

    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def Kmeans_clustering(X, k, pca_components, target_tickers, nr_most_similiar):
    
    X_pca = pca_transform(X, pca_components)

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_IDs = kmeans.fit_predict(X_pca)

    target_tickers = ["TSLA","HOOD","AMD"]
    evaluate_clustering(target_tickers,cluster_IDs,X_pca,"KMeans",nr_similar= nr_most_similiar)

def evaluate_clustering(target_tickers,cluster_IDs,X_pca,title,nr_similar = 5,skip_target_graph = False):
    for target_ticker in target_tickers:
        target_index = X_pca.index.get_loc(target_ticker)
        my_cluster_Id = cluster_IDs[target_index]
        
        # Find all indices in the same cluster
        cluster_members_indices = []
        for i, cluster_id in enumerate(cluster_IDs):
            if cluster_id == my_cluster_Id:
                if i != target_index:
                    cluster_members_indices.append(i)

        if len(cluster_members_indices) == 0:
            nearest_tickers = []
        else:
            dists = cosine_distances([X_pca.iloc[target_index]], X_pca.iloc[cluster_members_indices])[0]
            
            sorted_tickers = [x for _, x in sorted(zip(dists, X_pca.iloc[cluster_members_indices].index))]
            nearest_tickers = sorted_tickers[:nr_similar]
            
        if len(nearest_tickers) > 0:
            if skip_target_graph:
                plot_candlestick (nearest_tickers,[f" Rank{i}" for i in range(1, nr_similar+1)],title,nr_cols=3)
                continue
            plot_candlestick ([target_ticker],['Target'],title,nr_cols=3)    
            plot_candlestick (nearest_tickers,[f" Rank{i}" for i in range(1, nr_similar+1)],title,nr_cols=3)
            

#Agglomerative clustering function###########################################################################################
def Agglomerative_clustering(X_pca, target_tickers, nr_most_similiar):
    model = AgglomerativeClustering(
        distance_threshold=10,
        n_clusters=None, 
        metric='cosine',
        linkage='average'
    )
    cluster_IDs = model.fit_predict(X_pca)
    evaluate_clustering(target_tickers,cluster_IDs,X_pca,"Agglo",nr_similar= nr_most_similiar)
    
def Agglomerative_clustering_hyper_tune(X,X_pca, target_tickers, nr_most_similiar,pca_components):
    linkages = [
        'ward',
        'average',
        'complete',
        'single',   
    ]

    metrics = [
        'euclidean',
        'manhattan',
        'l1',
        'l2',
        'chebyshev',
        'cosine',
    ]

    for pca_component in pca_components:
        for _linkage in linkages:
            for _metric in metrics:
                
                if _linkage == 'ward' and _metric != 'euclidean':
                    continue
                
                X_pca = pca_transform(X, pca_component)

                model = AgglomerativeClustering(
                    distance_threshold=10,
                    n_clusters=None, 
                    metric=_metric,
                    linkage=_linkage
                )
                cluster_IDs = model.fit_predict(X_pca)
                title = f"Agglo-{_linkage}-{_metric}-PCA{pca_component}"
                evaluate_clustering(target_tickers,cluster_IDs,X_pca,title,nr_similar= nr_most_similiar,skip_target_graph=True)