# Consumer Complaint Classification using BERT

This project implements a BERT-based text classification system for categorizing consumer complaints into different product categories. The system uses the BERT (Bidirectional Encoder Representations from Transformers) model to analyze complaint narratives and predict the appropriate product category.

## Project Structure

- `app.py`: Main application for training and evaluating the BERT model
- `predict.py`: Script for making predictions on new complaint texts
- `bert_model/`: Directory containing the trained BERT model (not included in repository)
- `complaints_processed.csv`: Dataset of consumer complaints (not included in repository)

## Features

- BERT-based text classification
- GPU acceleration support
- Interactive prediction interface
- Model evaluation with confusion matrix visualization
- Confidence scores for predictions

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- pandas
- scikit-learn
- matplotlib
- tqdm

Install the required packages using:

```bash
pip install torch transformers pandas scikit-learn matplotlib tqdm
```

## Usage

### Training the Model

To train the model, run:

```bash
python app.py
```

This will:

1. Load the complaints dataset
2. Train a BERT model if no existing model is found
3. Evaluate the model's performance
4. Save the trained model to the `bert_model/` directory

### Making Predictions

To make predictions on new complaint texts, run:

```bash
python predict.py
```

This will start an interactive session where you can enter complaint texts and get predictions along with confidence scores.

Example usage:

```
Enter your complaint text (type 'quit' to exit):

Complaint: I was charged twice for a transaction I didn't authorize, and the bank won't help.

Predicted Product: Credit card
Confidence: 85.23%
```

## Model Details

- Base Model: BERT (bert-base-uncased)
- Maximum sequence length: 128 tokens
- Training epochs: 3
- Learning rate: 2e-5
- Batch size: 16

## Performance

The model's performance is evaluated using:

- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Validation accuracy and loss metrics

## Notes

- The model requires a GPU for optimal performance but will fall back to CPU if no GPU is available
- The training process may take several hours depending on your hardware
- Make sure to have sufficient disk space for the model files and dataset
