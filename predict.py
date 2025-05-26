import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BERTPredictor:
    def __init__(self, model_path='./bert_model'):
        print(f"Loading model from {model_path}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        print("Model loaded successfully!")

    def predict(self, text):
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move input to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction].item()

        return prediction.item(), confidence

def main():
    # Load label mapping from the original training data
    df = pd.read_csv("complaints_processed.csv")
    unique_products = df['product'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_products)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Initialize predictor
    predictor = BERTPredictor()

    # Interactive prediction loop
    print("\nEnter your complaint text (type 'quit' to exit):")
    while True:
        text = input("\nComplaint: ").strip()
        if text.lower() == 'quit':
            break

        if text:
            pred_idx, confidence = predictor.predict(text)
            pred_label = idx_to_label[pred_idx]
            print(f"\nPredicted Product: {pred_label}")
            print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 