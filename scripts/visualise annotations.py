import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

prefix = ".."
path = prefix + "/data/wallets_to_annotate_manual.xlsx"
df_excel = pd.read_excel(path)
df = df_excel[["fromAddress", "blockNumber", "label"]]

# create graph

df_agg = df.groupby(["blockNumber", "label"]).count()

# create a node for each label use the number of wallets in df_agg as the size
# create an edge for each blockNumber

G = nx.Graph()
blocks = list(df_agg.index.get_level_values(0).unique())
i = 0
for index, row in df_agg.iterrows():
    i+=1
    print(row)
    G.add_node(i, label=index[1], size=(1000000*row["fromAddress"])**(1/2))
    G.add_edge(index[0], i)
    #aggreate on the blockNumber for size
    G.add_node(index[0], label= index[0], size=(1000000*df_agg.loc[index[0]].sum()["fromAddress"])**(1/2))

G.add_node("blocks", label="blocks", size=(1000000*len(df))**(1/2))
G.add_edges_from([("blocks", node) for node in blocks if node != "blocks"])

labels = nx.get_node_attributes(G, 'label')
# draw beautiful graph
plt.figure(figsize=(20,20))


pos = nx.spring_layout(G, seed=31137465)  # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[node]["size"] for node in G.nodes], node_color="lightblue")
nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color="grey")
nx.draw_networkx_labels(G, pos, labels, font_size=20)
plt.show()

for node in G.nodes:
    print(node, G.nodes[node])

df_agg = df.groupby(["label"]).count()
df_agg = df_agg.sort_values(by="fromAddress", ascending=False)["fromAddress"]
# tight layout
plt.figure(figsize=(20,20))
plt.barh(df_agg.index, df_agg.values)
plt.yticks(fontsize=20)
plt.title("Number of wallets per label", fontsize=40)
plt.show()

df