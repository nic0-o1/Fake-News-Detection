# Fake News Detection

This repository contains three Jupyter notebooks designed for the Fake News Detection project. These notebooks follow a systematic approach to identifying and analyzing fake news, from data preprocessing to model training and results visualization. Below is an overview of each notebook and its purpose.

## Notebooks Overview

### 0. Data Exploration and Processing

**File:** `0_DataExplorationProcessing.ipynb`

This notebook is dedicated to the initial stages of data exploration and preprocessing. It includes the following steps:
- Loading and inspecting the dataset.
- Cleaning the text data by removing duplicates, non-English articles, and other irrelevant content.
- Generating word clouds to visually compare true and fake news.
- Saving the cleaned and processed dataset for later use.

### 1. Models Training

**File:** `1_ModelsTraining.ipynb`

This notebook is focused on training Long Short-Term Memory (LSTM) models for fake news classification, with and without attention mechanisms. Key steps include:
- Splitting the dataset into training, validation, and testing sets.
- Initializing and training LSTM models.
- Evaluating model performance on various metrics.
- Saving the training results, including model weights and evaluation metrics.

### 2. Results Visualization

**File:** `2_ResultsVisualization.ipynb`

This notebook is used to visualize and interpret the results of the trained models. It includes:
- Loading saved training results and model metrics.
- Generating visualizations of metrics such as loss, accuracy, precision, recall, and F1-score.
- Comparing the performance of models to draw insights.

## Getting Started

To get started with the project, follow these instructions:

1. Clone the repository:
    ```bash
    git clone https://github.com/nic0-o1/Fake-News-Detection.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Fake-News-Detection/src
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Open the notebooks in Jupyter:
    ```bash
    jupyter notebook
    ```

## Usage

1. **Data Exploration and Processing:**
   - Open the notebook `0_DataExplorationProcessing.ipynb`.
   - Run all cells to preprocess the dataset and save the processed data.

2. **Models Training:**
   - Open the notebook `1_ModelsTraining.ipynb`.
   - Run all cells to train the LSTM models and save the training results.

3. **Results Visualization:**
   - Open the notebook `2_ResultsVisualization.ipynb`.
   - Run all cells to visualize the performance metrics of the trained models. (You can also load previous results from `checkpoints/results.json`).

## Additional Information

- **Model Checkpoints:** The models and results are saved in the `checkpoints/` directory.
