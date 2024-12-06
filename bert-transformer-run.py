import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import *
from model import TransformerModel

def train(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(train_loader, desc="Training", ncols=100, leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        loss.backward()
        optimizer.step()

        avg_loss = total_loss / (len(progress_bar) + 1)
        accuracy = correct_predictions.double() / total_predictions
        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy.item())

        # Log to wandb
        # wandb.log({"train_loss": avg_loss, "train_accuracy": accuracy.item(), "epoch": epoch})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions.double() / total_predictions
    return avg_loss, accuracy


def evaluate(model, val_loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(val_loader, desc="Evaluating", ncols=100, leave=False)

    with torch.no_grad():
        for batch in progress_bar:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            avg_loss = total_loss / (len(progress_bar) + 1)
            accuracy = correct_predictions.double() / total_predictions
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy.item())

            # Log to wandb
            # wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy.item(), "epoch": epoch})

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions.double() / total_predictions
    return avg_loss, accuracy

def run_epochs(num_epochs, model, train_loader, val_loader, optimizer, loss_fn, device, scheduler):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device, epoch)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}")
        
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, epoch)
        print(f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        # 如果当前验证准确率高于最佳值，保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_checkpoint.pth')
            print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

def test(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(test_loader, desc="Testing", ncols=100, leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            avg_loss = total_loss / (len(progress_bar) + 1)
            accuracy = correct_predictions.double() / total_predictions
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy.item())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions.double() / total_predictions
    print(f"Test loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    MAX_SEQ_LEN = 300
    vocab_size = tokenizer.vocab_size
    epochs = 50
    use_standard_dataset = False

    # Create DataLoader for training, validation, and testing
    train_dataset_path = "train.csv"
    val_dataset_path = "val.csv"
    test_dataset_path = "test.csv"

    if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) and os.path.exists(test_dataset_path):
        print("Loading train, val, and test datasets")
        train_df = pd.read_csv(train_dataset_path)
        val_df = pd.read_csv(val_dataset_path)
        test_df = pd.read_csv(test_dataset_path)
        batch_size = 256 if use_standard_dataset else 32
    else:
        print("Preprocessing dataset")
        if use_standard_dataset:
            batch_size = 256
            whole_df = pd.read_csv("WeiboSentiment2019.csv", header=None, names=["index", "label", "tweet"]).drop(columns=["index"])
            whole_df = whole_df.shift(-1).dropna().reset_index(drop=True)
        else:
            batch_size = 32
            whole_df = pd.read_csv("jp_dataset.csv", header=None, names=["time", "tweet", "label"], encoding='GBK').drop([0]).reset_index(drop=True)
            whole_df = whole_df[whole_df['label']!='2'].reset_index(drop=True)
            whole_df['label'] = whole_df['label'].astype(int)
        
        train_df = whole_df.sample(frac=0.8, random_state=42)
        val_test_df = whole_df.drop(train_df.index)
        val_df = val_test_df.sample(frac=0.5, random_state=42)
        test_df = val_test_df.drop(val_df.index)
        train_df.to_csv(train_dataset_path, index=False)
        val_df.to_csv(val_dataset_path, index=False)
        test_df.to_csv(test_dataset_path, index=False)
        print("Finish preprocessing dataset")

    if use_standard_dataset:
        train_dataset = WeiboSentimentDataset(train_df, tokenizer, max_len=MAX_SEQ_LEN)
        val_dataset = WeiboSentimentDataset(val_df, tokenizer, max_len=MAX_SEQ_LEN)
        test_dataset = WeiboSentimentDataset(test_df, tokenizer, max_len=MAX_SEQ_LEN)
    else:
        train_dataset = WeiboDataset(pd.concat([train_df, val_df]), tokenizer, max_len=MAX_SEQ_LEN)
        val_dataset = WeiboDataset(val_df, tokenizer, max_len=MAX_SEQ_LEN)
        test_dataset = WeiboDataset(test_df, tokenizer, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TransformerModel(vocab_size=vocab_size, use_pretrained_bert=False, freeze_bert=True).to(device)
    print(f'model parameters (unfreezed): {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'model parameters (total): {sum(p.numel() for p in model.parameters())}')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    run_epochs(epochs, model, train_loader, val_loader, optimizer, loss_fn, device, scheduler)

    # After training, evaluate the model on the test set
    test_loss, test_accuracy = test(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")

    # Log final test results to wandb
    # wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    # Finish wandb run
    # wandb.finish()
