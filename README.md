# BERT Fine-Tuning for Sentiment Classification

## Overview
This project implements a sentiment classification model using **BERT** (Bidirectional Encoder Representations from Transformers). The model is fine-tuned on a Twitter dataset, where the task is to classify tweets into three categories: **positive**, **neutral**, and **negative** sentiment.

The project involves multiple stages:
1. Dataset preparation and custom data loading.
2. Fine-tuning the **DistilBERT** model, a lightweight version of BERT.
3. Comparing the performance of models trained from scratch and pre-trained models.
4. Evaluating model performance using accuracy metrics.

## Project Structure
- **Data**: 
  - The dataset consists of 5,000 tweets, labeled as positive, neutral, or negative. The data is split into training (3,000 tweets), validation (1,000 tweets), and test sets (1,000 tweets).
  - The dataset can be found in `Tweets_5K.csv`.

- **Main Concepts**:
  - Preprocessing tweets with tokenization.
  - Fine-tuning BERT using Hugging Face's Transformers library.
  - Implementing custom PyTorch datasets and data loaders.
  - Evaluating performance through accuracy and visualizations.

## Tasks

### 1. Dataset Preparation
- The dataset is loaded and split into training, validation, and test sets.
- Each tweet is assigned a label: 
  - **0** for positive
  - **1** for neutral
  - **2** for negative.
  
- **Custom PyTorch Dataset**: A custom dataset class is implemented to preprocess the tweets using the BERT tokenizer. The input format required by BERT includes special tokens ([CLS], [SEP]) and an attention mask to indicate non-padding tokens.

### 2. Building the Model
- **Model Architecture**: 
  - We use **DistilBERT**, a smaller and faster variant of BERT, specifically `distilbert-base-uncased`.
  - The model is fine-tuned for sentiment classification using a sequence classification head with 3 output classes.
  
- **Model Versions**:
  - **Untrained Model**: An untrained DistilBERT model is evaluated to establish a baseline.
  - **Manually Trained Model**: The untrained model is trained from scratch using the sentiment classification dataset.
  - **Pre-trained Model**: The pre-trained DistilBERT model is evaluated without fine-tuning to observe its initial performance.
  - **Fine-tuned Model**: The pre-trained DistilBERT model is fine-tuned for the sentiment classification task over two epochs.

### 3. Training and Evaluation
- **Training**: 
  - The model is trained using the **AdamW** optimizer with a learning rate of 5e-5.
  - The training process includes a **learning rate scheduler** for gradual learning rate adjustments.
  - The model is trained for **2 epochs** and evaluated after each epoch on both the training and validation sets.

- **Evaluation Metrics**: The primary evaluation metric is **accuracy**, calculated after each epoch and on the final test set.

### 4. Results
- **Baseline Accuracy**: The untrained model achieves an accuracy of **26.5%** on the test set.
- **Manually Trained Model**: After two epochs, the manually trained model achieves an accuracy of **57.9%**.
- **Pre-trained Model**: Without fine-tuning, the pre-trained model achieves a baseline accuracy of **27.3%**.
- **Fine-tuned Model**: After fine-tuning the pre-trained model for two epochs, the accuracy improves to **75.6%**.

### 5. Visualizations
- **Performance Comparison**: A bar plot is generated to compare the performance of the baseline, manually trained, pre-trained, and fine-tuned models.

## Requirements
- Python 3.x
- Libraries:
  - `transformers`
  - `torch`
  - `datasets`
  - `tqdm`
  - `scikit-learn`
  - `matplotlib`
  
## How to Run

1. Install the required libraries:
  
2. Download the dataset:
    - The dataset `Tweets_5K.csv` is available from the provided URL or can be uploaded manually.

3. Run the training and evaluation pipeline:
    - Use the provided Python code to preprocess the dataset, fine-tune the model, and evaluate its performance.

4. Visualize the results:
    - Accuracy metrics for each model version (baseline, manually trained, pre-trained, fine-tuned) will be displayed in a bar plot.

## Results
- **Baseline Accuracy**: 26.5%
- **Manually Trained Accuracy**: 57.9%
- **Pre-trained Model Accuracy**: 27.3%
- **Fine-tuned Model Accuracy**: 75.6%

## Acknowledgments
This project utilizes the following resources:html#automodelforsequenceclassification

2. https://stackoverflow.com/questions/65097733/creating-a-train-and-a-test-dataloader

3. https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

4. https://pytorch.org/docs/stable/generated/torch.argmax.html

5. https://huggingface.co/docs/transformers/en/model_doc/distilbert

6. https://www.kdnuggets.com/2023/03/introduction-getitem-magic-method-python.html
