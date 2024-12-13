# Temporal Graph Neural Networks for Stock Return Prediction

This project explores the application of Graph Neural Networks (GNNs) for stock return prediction, leveraging both temporal and relational dependencies in financial data. Using the S&P 500 dataset, we compare window-based GNNs and temporal GNNs, integrating LSTMs to capture historical trends. Graph construction approaches—sector-independent, fully connected, and hypergraphs—are evaluated to study inter-sector relationships. Results show that temporal GNNs with fully connected graphs and learnable correlation weights outperform other methods, achieving a test Sign Accuracy of 83.1%. This highlights the importance of inter-sector dependencies and the integration of temporal-spatial modeling for robust stock return prediction.