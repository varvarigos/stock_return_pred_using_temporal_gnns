# Temporal Graph Neural Networks for Stock Return Prediction

This project explores the application of Graph Neural Networks (GNNs) for stock return prediction, leveraging both temporal and relational dependencies in financial data. Using the S&P 500 dataset, we compare window-based GNNs and temporal GNNs, integrating LSTMs to capture historical trends. Graph construction approaches—sector-independent, fully connected, and hypergraphs—are evaluated to study inter-sector relationships. Results show that temporal GNNs with fully connected graphs and learnable correlation weights outperform other methods, achieving a test Sign Accuracy of 83.1%. This highlights the importance of inter-sector dependencies and the integration of temporal-spatial modeling for robust stock return prediction.

### How to Run the Code

To reproduce the results from the project:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/varvarigos/stock_return_pred_using_temporal_gnns.git
   ```
2. Move all the files to the same directory (if not already organized).
3. Navigate to the directory in your terminal:
   ```bash
   cd stock_return_pred_using_temporal_gnns
   ```
4. Set up the Python environment using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
5. Open the `project_code.ipynb` notebook.
6. Press **Run All** to execute all the cells. 

The hyperparameters are pre-configured to reproduce the best-performing model as described in the report. 
