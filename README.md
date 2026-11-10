# Facebook Influence Maximization & Social Network Analysis

Analysis of a large-scale Facebook social network using centrality measures, information diffusion, influence maximization, and community detection.

## Project Overview

This academic project analyzes a Facebook social network and investigates how different strategies can be used to identify influential users and maximize information diffusion.

## Objectives

- Analyze the structure of a social network.
- Compare different centrality measures for identifying influential users.
- Simulate information diffusion using the Independent Cascade model.
- Compare different influence-maximization strategies.
- Analyze communities within the network.

## Methodology

The project includes:

1. Network construction and analysis.
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

## Technologies

- Python
- NetworkX
- Pandas
- NumPy
- Matplotlib

## Evaluation

The strategies were evaluated using information diffusion and network coverage.

The reported experiments compared the influence achieved by different seed-selection strategies under the Independent Cascade model.

## Project Structure

```text
facebook-influence-maximization/
│
├── README.md
└── ...
