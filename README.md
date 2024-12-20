# Clustering of Image Data Set Using K-Means and Fuzzy K-Means Algorithms

This repository contains the implementation of the research paper **"Clustering of Image Data Set Using K-Means and Fuzzy K-Means Algorithms"** by **Vinod Kumar Dehariya, Shailendra Kumar Shrivastava, and R. C. Jain**. The project demonstrates clustering techniques applied to image data segmentation using Python.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Overview
Clustering is a fundamental technique in image processing, data analysis, and computer vision. This project compares the performance of **K-Means** and **Fuzzy K-Means** algorithms in clustering image datasets for segmentation. The results highlight the effectiveness of fuzzy logic in improving clustering performance.

## Features
- Implementation of **K-Means** and **Fuzzy K-Means** algorithms.
- Image segmentation using clustering techniques.
- Visualization of clustering results and centroids.
- Analysis of algorithm performance based on computation time and accuracy.

## Algorithms
### K-Means Algorithm
- Assigns each data point to the nearest cluster based on the Euclidean distance.
- Iteratively updates cluster centroids until convergence.

### Fuzzy K-Means Algorithm
- Allows each data point to belong to multiple clusters with a degree of membership.
- Incorporates fuzzy logic to improve cluster boundaries.

Both algorithms are implemented for clustering image datasets to demonstrate segmentation accuracy.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shahzan01/Pattern-recognition.git
   cd Pattern-recognition

## Usage
- Place your image datasets in the data/ folder.
Run the main script to perform clustering:
   ```bash
   python main.py
- Output results, such as clustered images and visualizations, will be saved in the output/ directory.
## Results
- Centroid Movement: The movement of centroids during the clustering process is visualized.
- Clustering Visualization: Both K-Means and Fuzzy K-Means clustering results are displayed for comparison.
- Improved Segmentation with Fuzzy Factor: Fuzzy K-Means provides smoother and more accurate segmentation by allowing degrees of membership.
## Example Outputs:
- Original Image:

- K-Means Clustering Result:

- Fuzzy K-Means Clustering Result:

