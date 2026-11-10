import networkx as nx
import pandas as pd

print("Loading Facebook dataset...")

G = nx.read_edgelist(
    "facebook_combined.txt",
    nodetype=int
)

print("Dataset loaded")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nCalculating centralities...")

degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, k=100, seed=42)
closeness = nx.closeness_centrality(G)

results = pd.DataFrame({
    "Node": list(G.nodes()),
    "Degree": [degree[n] for n in G.nodes()],
    "Betweenness": [betweenness[n] for n in G.nodes()],
    "Closeness": [closeness[n] for n in G.nodes()]
})

results = results.sort_values(
    by="Degree",
    ascending=False
)

print("\nTop 10 Influential Users")
print(results.head(10))

results.to_csv("facebook_influencers.csv", index=False)

print("\nSaved: facebook_influencers.csv")