# Facebook Influence Maximization & Social Network Analysis

Analysis of a large-scale Facebook social network using centrality measures, information diffusion, influence maximization, and community detection.

## Project Overview

This academic project analyzes a Facebook social network to identify influential users, simulate information diffusion, compare seed-selection strategies, and study community structure.

## Objectives

- Analyze the structure of a social network.
- Identify influential users using different centrality measures.
- Simulate information diffusion using the Independent Cascade model.
- Compare different seed-selection strategies.
- Study community structure using community detection methods.

## Methodology

The project includes:

1. Network construction and preprocessing.
2. Degree, betweenness, closeness, and eigenvector centrality analysis.
3. Independent Cascade diffusion simulation.
4. Comparison of random and centrality-based seed-selection strategies.
5. Greedy influence maximization.
6. Community detection and visualization.

## Strategies Compared

- Random
- Top Degree
- Top Betweenness
- Top Closeness
- Top Eigenvector
- Greedy Influence Maximization

## Evaluation

The strategies were evaluated using:

- **Influence:** number of users reached through simulated information diffusion.
- **Coverage:** percentage of the network reached by the diffusion process.

The experiments used the Independent Cascade model with a propagation probability of 0.05 and multiple Monte Carlo simulations.

## Results

The experiment compared six seed-selection strategies. The reported results showed:

| Strategy | Influence | Coverage |
|---|---:|---:|
| Top Degree | 2167 | 53.6% |
| Top Betweenness | 2163 | 53.5% |
| Greedy Influence Maximization | 2163 | 53.6% |
| Random | 2091 | 51.8% |
| Top Closeness | 2005 | 49.6% |
| Top Eigenvector | 1811 | 44.8% |

The results indicate that degree-based and greedy strategies achieved the highest diffusion performance among the evaluated approaches.

## Community Analysis

Community detection was used to identify groups of closely connected users within the network.

The analysis identified **10 communities**, with community sizes ranging from **37 to 983 users**.

## Visualizations

### Top 10 Influential Users

`Figure_1_deep.png`

### Strategy Comparison

`Figure_2_deep.png`

### Diminishing Returns

`Figure_3_deep.png`

### Community Visualization

`Figure_4_deep.png`

## Technologies

- Python
- NetworkX
- Pandas
- NumPy
- Matplotlib

## Project Structure

```text
facebook-influence-maximization/
│
├── facebook_final_project.py
├── analysis.py
├── data.py
├── strategy_comparison.csv
├── community_sizes.csv
│
├── Figure_1_deep.png
├── Figure_2_deep.png
├── Figure_3_deep.png
├── Figure_4_deep.png
│
└── README.md
```

## Academic Context

This project was developed as an academic study of social network analysis and influence maximization, with a focus on graph-based methods, information diffusion, and community detection.
