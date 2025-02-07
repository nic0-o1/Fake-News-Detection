# Fake News Detection

This folder contains three Jupyter notebooks that are part of the Fake News Detection project. Each notebook serves a specific purpose in the process of detecting fake news. Below is a brief explanation of each notebook:

## Notebooks Overview

### 0. Data Exploration and Processing

**File:** 0_DataExplorationProcessing.ipynb

This notebook is responsible for the initial data exploration and preprocessing steps. It includes:
- Loading the dataset
- Cleaning the text data (removing duplicates, non-English articles, etc.)
- Generating word clouds for true and fake news
- Saving the processed dataset

### 1. Models Training

**File:** 1_ModelsTraining.ipynb

This notebook focuses on training LSTM models with and without attention mechanisms. It includes:
- Splitting the dataset into training, validation, and testing sets
- Initializing and training the models
- Evaluating the models' performance
- Saving the training results

### 2. Results Visualization

**File:** 2_ResultsVisualization.ipynb

This notebook is used to visualize the results of the trained models. It includes:
- Loading the saved results
- Visualizing various metrics such as loss, accuracy, precision, recall, and F1-score

## Getting Started

To get started with the project, follow these steps:

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
   - Open 0_DataExplorationProcessing.ipynb
   - Run all cells to preprocess the dataset and save the processed data.

2. **Models Training:**
   - Open 1_ModelsTraining.ipynb
   - Run all cells to train the LSTM models and save the training results.

3. **Results Visualization:**
   - Open 2_ResultsVisualization.ipynb
   - Run all cells to visualize the performance metrics of the trained models (Previous results available at `checkpoints/results.json`).
