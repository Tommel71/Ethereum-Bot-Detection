import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn import metrics as skmetrics
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def metrics(clusters: pd.DataFrame, target: pd.DataFrame):
    """Calculate accuracy, recall, precision, and F1-score using scikit-learn."""

    # Assuming 'label' column contains the predicted labels in the 'clusters' DataFrame
    predicted_labels = clusters['label'].values

    # Assuming 'label' column contains the true labels in the 'target' DataFrame
    true_labels = target['label'].values


    # map labels to integers
    label_to_int = {label: i for i, label in enumerate(np.unique(true_labels))}
    true_labels = np.array([label_to_int[x] for x in true_labels])
    predicted_labels = np.array([label_to_int[x] for x in predicted_labels])

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate recall (sensitivity or true positive rate)
    recall = recall_score(true_labels, predicted_labels, average="macro")

    # Calculate precision (positive predictive value)
    precision = precision_score(true_labels, predicted_labels, average="macro")

    # Calculate F1-score
    f1 = f1_score(true_labels, predicted_labels, average="macro")

    return accuracy, recall, precision, f1

def eval_clustering(clusters: pd.DataFrame, target: pd.DataFrame):
    """
    clusters looks like this
    address, cluster_id

    target looks like this
    address, label

    This function tries all the possible cluster_id to target assigments and returns the best assignment as a dict
    and the best score as a float.

    Metric is the accuracy score.

    :param clusters:
    :param target:
    :return:
    """

    assert len(clusters) == len(target)
    clusters = clusters.copy()
    unique_cluster_ids = clusters['cluster_id'].unique()
    clusters["label"]= None
    # Iterate through all possible cluster ID assignments
    for cluster_id in unique_cluster_ids:

        mask = clusters['cluster_id'] == cluster_id
        target_cluster = target[mask]
        # because there could be issues with class imbalance
        max_label = target_cluster['label'].value_counts().idxmax()

        clusters["label"][mask] = max_label


    # Calculate the accuracy score for the best assignment
    accuracy, recall, precision, f1 = metrics(clusters, target)

    return accuracy, recall, precision, f1, clusters


def benchmark(clusters: pd.DataFrame, target: pd.DataFrame, k=1000):
    clusters = clusters.copy()
    scores = []
    for _ in tqdm(range(k)):
        clusters["cluster_id"] = clusters["cluster_id"].sample(frac=1).values
        scores.append(eval_clustering(clusters, target)[0])

    plt.hist(scores, bins=100)
    plt.show()
    # get confidence interval
    lower = np.quantile(scores, 0.025)
    upper = np.quantile(scores, 0.975)
    interval = (lower, upper)
    mean = np.mean(scores)

    return interval, mean, scores


def purity_scores_single(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = skmetrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.amax(contingency_matrix, axis=0) / np.sum(contingency_matrix, axis=0)


def entropy_scores_single(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cluster_ids = np.unique(y_pred)
    classes = np.unique(y_true)
    c = len(classes)
    k = len(cluster_ids)

    def single_entropy(cluster, target_in_cluster):
        values_to_sum = []
        for class_ in classes:

            if len(cluster) == 0:
                values_to_sum.append(0)
            else:

                ratio = len(cluster[target_in_cluster == class_]) / len(cluster)
                if ratio == 0:
                    values_to_sum.append(0)
                else:
                    values_to_sum.append(ratio * np.log(ratio))
        print(values_to_sum)
        print(c)
        return -1/np.log(c) * np.sum(values_to_sum)

    entropies = []
    sizes = []
    for i in range(k):
        mask = y_pred == cluster_ids[i]
        cluster = y_true[mask]
        target_in_cluster = y_true[mask]
        sizes.append(len(cluster))
        entropies.append(single_entropy(cluster, target_in_cluster))

    return np.array(entropies)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = skmetrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def entropy_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cluster_ids = np.unique(y_pred)
    classes = np.unique(y_true)
    c = len(classes)
    k = len(cluster_ids)

    def single_entropy(cluster, target_in_cluster):
        values_to_sum = []
        for class_ in classes:

            if len(cluster) == 0:
                values_to_sum.append(0)
            else:

                ratio = len(cluster[target_in_cluster == class_]) / len(cluster)
                if ratio == 0:
                    values_to_sum.append(0)
                else:
                    values_to_sum.append(ratio * np.log(ratio))
        print(values_to_sum)
        print(c)
        return -1/np.log(c) * np.sum(values_to_sum)

    entropies = []
    sizes = []
    for i in range(k):
        mask = y_pred == cluster_ids[i]
        cluster = y_true[mask]
        target_in_cluster = y_true[mask]
        sizes.append(len(cluster))
        entropies.append(single_entropy(cluster, target_in_cluster))


    # return weighted average of the entropies
    return np.sum(np.array(entropies) * np.array(sizes)) / np.sum(sizes)


if __name__ == "__main__":

    # make cluster_df have 100 rows and 10 clusters
    cluster_sizes = [11,13,43,12, 7, 14]
    clusterids = list(itertools.chain.from_iterable([[i]*cluster_sizes[i] for i in range(len(cluster_sizes))]))
    cluster_df = pd.DataFrame({"address": [f"0x{i}" for i in range(100)],
                                "cluster_id": clusterids})

    target_df = pd.DataFrame({"address": [f"0x{i}" for i in range(100)],
                                "label": [1 if i < 57 else 0 for i in range(100)]})


    print(eval_clustering(cluster_df, target_df))
    # we have cluster_df and now want to randomly permute the indices of the cluster_ids

    benchmark(cluster_df, target_df)


    purity_score([2,2,3, 1,1,1,1,1], [1,1,1,2,2,2, 2,2]) #  should be 0.875 when weighted 0.8333 when not weighted, we want weighted
    entropy_score([2,2,3, 1,1,1,1,1], [1,1,1,2,2,2, 2,2])
    entropy_score([1,2,3,1,2,3], [1,1,1,1,1,1]) # should be 1
    entropy_score([2,2,2,2,2,2], [1,1,1,1,1,1]) # gives nan because its not designed for just 1 class
    entropy_score([1,1,1,2,2,2], [3,3,3,1,1,1])
