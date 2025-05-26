import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Debug GPU information
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("No GPU available. Using CPU instead.")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, num_labels, model_name='bert-base-uncased', learning_rate=2e-5):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_labels = num_labels

    def save_model(self, path='./bert_model'):
        """Save model and tokenizer to specified path"""
        print(f"Saving model to {path}...")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print("Model saved successfully!")

    @classmethod
    def load_model(cls, path='./bert_model'):
        """Load model and tokenizer from specified path"""
        print(f"Loading model from {path}...")
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path)
        classifier = cls(num_labels=model.num_labels)
        classifier.model = model.to(device)
        classifier.tokenizer = tokenizer
        print("Model loaded successfully!")
        return classifier

    def train(self, train_loader, val_loader, num_epochs=3):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        print(f"Training on device: {self.device}")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Validation
            val_loss, val_accuracy = self.evaluate(val_loader)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += len(labels)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions.double() / total_predictions
        return avg_loss, accuracy

    def predict(self, texts):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, pred = torch.max(outputs.logits, dim=1)
                predictions.append(pred.item())
        
        return predictions

def main():
    print("Loading data...")
    df = pd.read_csv("complaints_processed.csv")
    df = df.dropna(subset=['narrative', 'product'])
    
    # Convert labels to numeric
    unique_products = df['product'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_products)}
    df['label'] = df['product'].map(label_to_idx)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['narrative'], 
        df['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Check if model exists
    model_path = './bert_model'
    if os.path.exists(model_path):
        print("\nLoading existing model...")
        model = BERTClassifier.load_model(model_path)
    else:
        print("\nTraining new model...")
        # Initialize tokenizer and create datasets
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer)
        test_dataset = TextDataset(X_test.tolist(), y_test.tolist(), tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Initialize and train model
        print("\nInitializing BERT model...")
        model = BERTClassifier(num_labels=len(unique_products))
        
        print("\nTraining BERT model...")
        model.train(train_loader, test_loader, num_epochs=3)
        
        # Save the trained model
        model.save_model()
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test.tolist())
    
    # Convert predictions back to original labels
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_pred_labels = [idx_to_label[pred] for pred in y_pred]
    y_test_labels = [idx_to_label[label] for label in y_test]
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=unique_products)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_products)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
    # Example prediction
    def predict_with_confidence(text):
        pred = model.predict([text])[0]
        label = idx_to_label[pred]
        return label
    
    example = "I was charged twice for a transaction I didn't authorize, and the bank won't help."
    label = predict_with_confidence(example)
    print(f"\nExample Prediction: {label}")

if __name__ == '__main__':
    main()
