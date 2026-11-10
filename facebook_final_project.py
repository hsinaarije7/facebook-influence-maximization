import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from networkx.algorithms.community import greedy_modularity_communities

print("=" * 80)
print("FACEBOOK INFLUENCE MAXIMIZATION - WITH VISUALIZATION")
print("=" * 80)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ==================================================
# 1. LOAD DATA - USE LARGEST COMPONENT ONLY
# ==================================================

G_full = nx.read_edgelist("facebook_combined.txt", nodetype=int)

print(f"Original graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

# Get largest connected component
largest_cc = max(nx.connected_components(G_full), key=len)
G = G_full.subgraph(largest_cc).copy()

print(f"Largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Discarded: {G_full.number_of_nodes() - G.number_of_nodes()} nodes")

# ==================================================
# 2. CENTRALITY CALCULATIONS
# ==================================================

print("\n" + "=" * 80)
print("CALCULATING CENTRALITY MEASURES")
print("=" * 80)

degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()), seed=42)
closeness = nx.closeness_centrality(G)

try:
    eigenvector = nx.eigenvector_centrality_numpy(G)
    print("✓ Eigenvector centrality calculated successfully")
except:
    eigenvector = degree.copy()
    print("! Using degree as eigenvector proxy")

# Results dataframe
results = pd.DataFrame({
    "Node": list(G.nodes()),
    "Degree": [degree[n] for n in G.nodes()],
    "Betweenness": [betweenness[n] for n in G.nodes()],
    "Closeness": [closeness[n] for n in G.nodes()],
    "Eigenvector": [eigenvector[n] for n in G.nodes()]
})
results = results.sort_values("Degree", ascending=False).reset_index(drop=True)

# ==================================================
# 3. FIGURE 1: TOP 10 INFLUENTIAL USERS
# ==================================================

top10 = results.head(10)

print("\n" + "=" * 80)
print("TOP 10 INFLUENTIAL USERS")
print("=" * 80)
print(top10[["Node", "Degree", "Betweenness", "Closeness"]].to_string(index=False))

plt.figure(figsize=(12, 6))
bars = plt.bar(top10["Node"].astype(str), top10["Degree"], color='steelblue', edgecolor='black')

plt.title("Top 10 Influential Facebook Users", fontsize=14, fontweight='bold')
plt.xlabel("User ID", fontsize=12)
plt.ylabel("Degree Centrality", fontsize=12)

for bar, val in zip(bars, top10["Degree"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.ylim(0, max(top10["Degree"]) * 1.1)
plt.tight_layout()
plt.savefig("Figure_1_deep.png", dpi=300, bbox_inches="tight")
plt.show()  # ← ADDED: This displays the figure
print("✓ Saved and displayed: Figure_1_deep.png")

# ==================================================
# 4. INDEPENDENT CASCADE MODEL
# ==================================================

def independent_cascade(G, seeds, p=0.05, iterations=100):
    """Independent Cascade diffusion model"""
    if not seeds:
        return 0
    
    total = 0
    for _ in range(iterations):
        activated = set(seeds)
        frontier = set(seeds)
        
        while frontier:
            next_frontier = set()
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor not in activated and random.random() < p:
                        next_frontier.add(neighbor)
            activated.update(next_frontier)
            frontier = next_frontier
        
        total += len(activated)
    
    return total / iterations

# ==================================================
# 5. GREEDY INFLUENCE MAXIMIZATION
# ==================================================

print("\n" + "=" * 80)
print("GREEDY INFLUENCE MAXIMIZATION")
print("=" * 80)

PROPAGATION_PROB = 0.05
NUM_SEEDS = 5
MC_ITERATIONS = 50

print(f"p = {PROPAGATION_PROB}, k = {NUM_SEEDS}, iterations = {MC_ITERATIONS}")

def greedy_im(G, k, p=0.05, iterations=50):
    seeds = []
    gains = []
    
    # Candidate pool: top 500 by degree
    candidates = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:500]]
    
    for i in range(k):
        best_node = None
        best_gain = 0
        current = independent_cascade(G, seeds, p, iterations) if seeds else 0
        
        for node in candidates:
            if node not in seeds:
                inf = independent_cascade(G, seeds + [node], p, iterations)
                gain = inf - current
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
        
        if best_node:
            seeds.append(best_node)
            gains.append(best_gain)
            print(f"  Seed {i+1}: Node {best_node} | Gain: +{best_gain:.0f}")
    
    return seeds, gains

greedy_seeds, marginal_gains = greedy_im(G, k=NUM_SEEDS, p=PROPAGATION_PROB, iterations=MC_ITERATIONS)

print(f"\nOptimal seeds: {greedy_seeds}")
print(f"Total influence: {sum(marginal_gains):.0f} nodes ({sum(marginal_gains)/G.number_of_nodes()*100:.1f}%)")

# ==================================================
# 6. FIGURE 2: STRATEGY COMPARISON
# ==================================================

print("\n" + "=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)

# Get seeds for each strategy
top_deg_seeds = results.nlargest(NUM_SEEDS, "Degree")["Node"].tolist()
top_bet_seeds = results.nlargest(NUM_SEEDS, "Betweenness")["Node"].tolist()
top_clo_seeds = results.nlargest(NUM_SEEDS, "Closeness")["Node"].tolist()
top_eig_seeds = results.nlargest(NUM_SEEDS, "Eigenvector")["Node"].tolist()
rand_seeds = random.sample(list(G.nodes()), NUM_SEEDS)

strategies = {
    "Random": rand_seeds,
    "Top Degree": top_deg_seeds,
    "Top Betweenness": top_bet_seeds,
    "Top Closeness": top_clo_seeds,
    "Top Eigenvector": top_eig_seeds,
    "Greedy (Ours)": greedy_seeds
}

results_comp = []
for name, seeds in strategies.items():
    inf = independent_cascade(G, seeds, p=PROPAGATION_PROB, iterations=100)
    cov = (inf / G.number_of_nodes()) * 100
    results_comp.append({
        "Strategy": name,
        "Influence": round(inf),
        "Coverage": round(cov, 1)
    })
    print(f"  {name:18s}: {inf:.0f} nodes ({cov:.1f}%)")

comp_df = pd.DataFrame(results_comp)
comp_df = comp_df.sort_values("Influence", ascending=False)

# FIGURE 2: Clean bar chart
plt.figure(figsize=(12, 7))

# Order for display
order = ["Random", "Top Degree", "Top Betweenness", "Top Closeness", "Top Eigenvector", "Greedy (Ours)"]
plot_data = []
for s in order:
    row = comp_df[comp_df["Strategy"] == s].iloc[0]
    plot_data.append(row)

colors = ['#1f77b4' if s != "Greedy (Ours)" else '#2ca02c' for s in order]
bars = plt.bar(range(len(order)), [d["Influence"] for d in plot_data], 
               color=colors, edgecolor='black', linewidth=1.5)

plt.title(f"Influence Maximization: Strategy Comparison (k={NUM_SEEDS}, p={PROPAGATION_PROB})", 
          fontsize=14, fontweight='bold')
plt.xlabel("Strategy", fontsize=12)
plt.ylabel("Expected Nodes Influenced", fontsize=12)

# Add labels on bars
for i, (bar, data) in enumerate(zip(bars, plot_data)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, 
             f"{data['Influence']}\n({data['Coverage']}%)", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(range(len(order)), order, rotation=45, ha='right', fontsize=11)
plt.ylim(0, max([d["Influence"] for d in plot_data]) * 1.15)
plt.grid(True, alpha=0.3, axis='y')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='Our Algorithm (Greedy)'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Baseline Strategies')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig("Figure_2_deep.png", dpi=300, bbox_inches="tight")
plt.show()  # ← ADDED: This displays the figure
print("✓ Saved and displayed: Figure_2_deep.png")

# ==================================================
# 7. FIGURE 3: DIMINISHING RETURNS
# ==================================================

plt.figure(figsize=(10, 6))

x = range(1, len(marginal_gains) + 1)

# Marginal gains bars
bars = plt.bar(x, marginal_gains, color='#d62728', alpha=0.7, edgecolor='black', label='Marginal Gain')

# Add line for cumulative
cumulative = np.cumsum(marginal_gains)
plt.plot(x, cumulative, marker='o', linewidth=2, markersize=10, 
         color='#1f77b4', label='Cumulative Influence')

plt.title("Diminishing Returns of Influence Maximization", fontsize=14, fontweight='bold')
plt.xlabel("Number of Seeds Added", fontsize=12)
plt.ylabel("Nodes Influenced", fontsize=12)
plt.xticks(x, x)
plt.grid(True, alpha=0.3, axis='y')

# Add labels on bars
for bar, val in zip(bars, marginal_gains):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'+{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add cumulative labels
for i, cum in enumerate(cumulative, 1):
    plt.annotate(f'Total: {cum:.0f}', (i, cum), 
                textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Figure_3_deep.png", dpi=300, bbox_inches="tight")
plt.show()  # ← ADDED: This displays the figure
print("✓ Saved and displayed: Figure_3_deep.png")

# ==================================================
# 8. FIGURE 4: COMMUNITY VISUALIZATION
# ==================================================

print("\n" + "=" * 80)
print("COMMUNITY DETECTION")
print("=" * 80)

communities = list(greedy_modularity_communities(G))
print(f"Found {len(communities)} communities")

# Get top 5 communities by size
top_comms = sorted(communities, key=len, reverse=True)[:5]

# Print community info
for i, comm in enumerate(top_comms, 1):
    top_influencer = max(comm, key=lambda x: degree[x])
    print(f"Community {i}: {len(comm)} nodes | Top influencer: {top_influencer}")

# Create subgraph of top 5 communities
nodes_sub = set()
for comm in top_comms:
    nodes_sub.update(comm)
subG = G.subgraph(nodes_sub)

plt.figure(figsize=(16, 12))

# Layout
pos = nx.spring_layout(subG, seed=42, k=0.5)

# Color by community
color_map = []
for node in subG.nodes():
    for i, comm in enumerate(top_comms):
        if node in comm:
            color_map.append(i)
            break

# Draw
nx.draw_networkx_nodes(subG, pos, node_color=color_map, node_size=100, 
                       cmap=plt.cm.Set3, alpha=0.9)
nx.draw_networkx_edges(subG, pos, alpha=0.15, width=0.5, edge_color='gray')

# Highlight greedy seeds
seeds_in_sub = [s for s in greedy_seeds if s in subG.nodes()]
if seeds_in_sub:
    nx.draw_networkx_nodes(subG, pos, nodelist=seeds_in_sub, 
                          node_color='red', node_size=350, 
                          edgecolors='black', linewidths=2.5,
                          label='Greedy Seeds')
    
    # Add seed labels
    seed_labels = {s: str(s) for s in seeds_in_sub}
    nx.draw_networkx_labels(subG, pos, seed_labels, font_size=10, font_weight='bold')

plt.title("Facebook Network Communities (Top 5 Communities)", fontsize=14, fontweight='bold')
plt.axis('off')
plt.legend(scatterpoints=1, loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig("Figure_4_deep.png", dpi=300, bbox_inches="tight")
plt.show()  # ← ADDED: This displays the figure
print("✓ Saved and displayed: Figure_4_deep.png")

# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

greedy_data = comp_df[comp_df["Strategy"] == "Greedy (Ours)"].iloc[0]
degree_data = comp_df[comp_df["Strategy"] == "Top Degree"].iloc[0]
random_data = comp_df[comp_df["Strategy"] == "Random"].iloc[0]

improvement = greedy_data["Influence"] - degree_data["Influence"]
improvement_pct = (improvement / degree_data["Influence"]) * 100

print(f"""
================================================================================
DATASET
================================================================================
Network size:        {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges
Network density:     {nx.density(G):.6f}
Average degree:      {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}

================================================================================
TOP INFLUENTIAL USER (Single Node)
================================================================================
User ID:             {int(top10.iloc[0]['Node'])}
Degree centrality:   {top10.iloc[0]['Degree']:.4f}
Betweenness:         {top10.iloc[0]['Betweenness']:.4f}
Closeness:           {top10.iloc[0]['Closeness']:.4f}

================================================================================
INFLUENCE MAXIMIZATION (k={NUM_SEEDS}, p={PROPAGATION_PROB})
================================================================================
Strategy            | Influence | Coverage | Improvement
--------------------|-----------|----------|-------------
Greedy (Ours)       | {greedy_data['Influence']:.0f}       | {greedy_data['Coverage']:.0f}%      | —
Top Degree          | {degree_data['Influence']:.0f}       | {degree_data['Coverage']:.0f}%      | +{improvement:.0f} ({improvement_pct:.1f}%)
Random              | {random_data['Influence']:.0f}       | {random_data['Coverage']:.0f}%      | +{greedy_data['Influence'] - random_data['Influence']:.0f}

Optimal seeds:       {greedy_seeds}

================================================================================
DIMINISHING RETURNS
================================================================================
Seed 1: +{marginal_gains[0]:.0f} nodes ({marginal_gains[0]/sum(marginal_gains)*100:.0f}% of total)
Seed 2: +{marginal_gains[1]:.0f} nodes
Seed 3: +{marginal_gains[2]:.0f} nodes
Seed 4: +{marginal_gains[3]:.0f} nodes
Seed 5: +{marginal_gains[4]:.0f} nodes

================================================================================
COMMUNITIES
================================================================================
Total communities:   {len(communities)}
Largest:             {len(top_comms[0])} nodes
Smallest:            {len(top_comms[-1])} nodes

================================================================================
CONCLUSION
================================================================================
The greedy algorithm and top-degree strategy both achieve {greedy_data['Influence']:.0f}
nodes ({greedy_data['Coverage']:.0f}% coverage), demonstrating that in this densely
connected Facebook network, degree centrality is an effective heuristic for
influence maximization.

================================================================================
FILES GENERATED
================================================================================
- Figure_1_deep.png     (Top 10 influencers)
- Figure_2_deep.png     (Strategy comparison with numbers)
- Figure_3_deep.png     (Diminishing returns)
- Figure_4_deep.png     (Community visualization)
""")

print("\n" + "=" * 80)
print("PROJECT COMPLETED! 4 figures should now be displayed on your screen.")
print("=" * 80)