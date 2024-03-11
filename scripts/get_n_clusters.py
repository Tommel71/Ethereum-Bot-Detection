import pandas as pd

def get_n_clusters_elbow(prefix):
    comb_alg = ["kmeans", "gmm"]
    comb_prepr = ["-1", "mean"]
    comb_dimred = ["UMAP", "non-UMAP"]

    data = []
    #prefix = "."
    for alg in comb_alg:
        for prepr in comb_prepr:
            for dimred in comb_dimred:
                name = f"{prefix}/outputs/large/cluster_results/evalset_elbow_clustered_nclusters_{alg}_{prepr}_{dimred}.csv"
                # read as txt and get int
                with open(name) as f:
                    n_clusters = f.read()
                    n_clusters = int(n_clusters)
                    data.append([alg, prepr, dimred, n_clusters])


    df = pd.DataFrame(data, columns=["alg", "prepr", "dimred", "n_clusters"])

    # map "gmm" to "GMM"
    df["alg"] = df["alg"].str.replace("gmm", "GMM")

    # set multiiindex
    df = df.set_index(["alg", "prepr", "dimred"])

    return df