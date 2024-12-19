import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import random
import os

# Ensure nltk resources are downloaded only once
import nltk
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Parameters
BATCH_SIZE = 128
EMBEDDING_DIM = 300
MAX_TITLE_LENGTH = 30
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_PERCENT_DATA = 0.01  # Flag to toggle between 20% or full dataset usage
DATA_COUNT = -1

# Load Train Dataset
def load_train_data(news_path, behaviors_path):
    news_df = pd.read_csv(news_path, sep='\t', header=None, 
                          names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                               names=["impression_id", "user_id", "time", "clicked_news", "impressions"])
    

    #behaviors_df = behaviors_df.sample(frac=USE_PERCENT_DATA, random_state=42).reset_index(drop=True)
    #print("Checking!!!!")
    #print(behaviors_df["clicked_news"])
    #print(behaviors_df.loc[:10,["impressions"]])
    return news_df, behaviors_df

# Load Validation Dataset
def load_validation_data(news_path, behaviors_path):
    news_df = pd.read_csv(news_path, sep='\t', header=None, 
                          names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                               names=["impression_id", "user_id", "time", "clicked_news", "impressions"])
    
    #behaviors_df = behaviors_df.sample(frac=USE_PERCENT_DATA, random_state=42).reset_index(drop=True)
    
    return news_df, behaviors_df



# Load Test Dataset
def load_test_data(news_path, behaviors_path):
    news_df = pd.read_csv(news_path, sep='\t', header=None, 
                          names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                               names=["impression_id", "user_id", "time", "clicked_news"])
    
    #behaviors_df = behaviors_df.sample(frac=USE_PERCENT_DATA, random_state=42).reset_index(drop=True)

    return news_df, behaviors_df


# Preprocess News Titles
def preprocess_news(news_df):
    news_df['tokenized_title'] = news_df['title'].apply(lambda x: word_tokenize(str(x).lower())[:MAX_TITLE_LENGTH])
    return news_df

# Generate Word Index Mapping
def build_vocab(news_df):
    vocab = set()
    for tokens in news_df['tokenized_title']:
        vocab.update(tokens)
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 1-based index
    word2idx['<PAD>'] = 0
    return word2idx

# Convert Titles to Indices
def titles_to_indices(news_df, word2idx):
    indices = []
    for tokens in news_df['tokenized_title']:
        indices.append([word2idx.get(w, 0) for w in tokens] + [0] * (MAX_TITLE_LENGTH - len(tokens)))
    news_df['title_indices'] = indices
    return news_df

# NRMS Model
class NRMS(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super(NRMS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, titles):
        embeds = self.embedding(titles)  # Shape: (batch_size, max_len, embed_dim)
        attn_output, _ = self.multihead_attn(embeds, embeds, embeds)
        attn_pooled = torch.mean(attn_output, dim=1)  # Pooling
        logits = self.fc(attn_pooled)
        return logits

# Dataset Class
class NewsDataset(Dataset):
    def __init__(self, titles, labels):
        self.titles = titles
        self.labels = labels

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return self.titles[idx], self.labels[idx]

# Train Function
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_titles, batch_labels in tqdm(train_loader):
        batch_titles, batch_labels = batch_titles.to(DEVICE), batch_labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_titles).squeeze(1)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluate Function
def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_titles, batch_labels in val_loader:
            batch_titles = batch_titles.to(DEVICE)
            outputs = torch.sigmoid(model(batch_titles)).squeeze(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    return all_preds, all_labels


# Prepare Training Dataset with Candidate News and Labels
def prepare_train_data(train_behaviors_df, train_news_df):

    if DATA_COUNT != -1:
        train_behaviors_df = train_behaviors_df.head(DATA_COUNT)

    train_titles = []
    train_labels = []

    for _, row in train_behaviors_df.iterrows():
        if pd.isna(row["impressions"]):
            continue  # Skip rows with missing impressions
        impressions = row["impressions"].split(" ")
        clicked_news = set(row["clicked_news"].split() if pd.notna(row["clicked_news"]) else [])
        #print()
        #print("Clicked News and Impression")
        #print(clicked_news)
        #print()
        #print()
        #print(impressions)

        for imp in impressions:
            #print(imp)
            if "-" in imp:
                news_id, click_flag = imp.split("-")
            matched_news = train_news_df[train_news_df["news_id"] == news_id]
            if not matched_news.empty:
                train_titles.append(matched_news["title_indices"].values[0])
                train_labels.append(1 if news_id in clicked_news else 0)

        #print(impressions)
        #print()
        #print()


    train_titles = torch.tensor(train_titles, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.float)

    return train_titles, train_labels

# Prepare Validation Dataset with Candidate News and Labels
def prepare_val_data(val_behaviors_df, val_news_df):
    
    if DATA_COUNT != -1:
        val_behaviors_df = val_behaviors_df.head(DATA_COUNT)

    val_titles = []
    val_labels = []

    for _, row in val_behaviors_df.iterrows():
        if pd.isna(row["impressions"]):
            continue  # Skip rows with missing impressions
        impressions = row["impressions"].split(" ")
        clicked_news = set(row["clicked_news"].split() if pd.notna(row["clicked_news"]) else [])


        for news_id in impressions:
            matched_news = val_news_df[val_news_df["news_id"] == news_id]
            if not matched_news.empty:
                val_titles.append(matched_news["title_indices"].values[0])
                val_labels.append(1 if news_id in clicked_news else 0)

    
    val_titles = torch.tensor(val_titles, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.float)

    return val_titles, val_labels




# Prepare Testing Dataset with Candidate News and Labels
def prepare_test_data(test_behaviors_df, test_news_df):

    if DATA_COUNT != -1:
        test_behaviors_df = test_behaviors_df.head(DATA_COUNT)

    test_titles = []
    test_labels = []

    for _, row in test_behaviors_df.iterrows():
        clicked_news = set(row["clicked_news"])

        for news_id in clicked_news:
            matched_news = test_news_df[test_news_df["news_id"] == news_id]
            if not matched_news.empty:
                test_titles.append(matched_news["title_indices"].values[0])
                test_labels.append(1)  # All clicked news are positive labels

    
    test_titles.append(torch.tensor(test_titles, dtype=torch.long))
    test_labels.append(torch.tensor(test_labels, dtype=torch.float))

    return test_titles, test_labels


# Test Function with Candidate News
def test_model_with_candidates(model, user_impressions, labels):
    model.eval()
    all_auc, all_mrr, all_ndcg_5, all_ndcg_10 = [], [], [], []

    with torch.no_grad():
        for candidates, true_labels in zip(user_impressions, labels):
            candidates = candidates.to(DEVICE)
            scores = model(candidates).squeeze(1).cpu().numpy()

            # Calculate Metrics for Each User
            auc, mrr, ndcg_5 = calculate_metrics(true_labels.tolist(), scores, k=5)
            _, _, ndcg_10 = calculate_metrics(true_labels.tolist(), scores, k=10)

            all_auc.append(auc)
            all_mrr.append(mrr)
            all_ndcg_5.append(ndcg_5)
            all_ndcg_10.append(ndcg_10)

    return np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg_5), np.mean(all_ndcg_10)

# Calculate Metrics
def calculate_metrics(y_true, y_scores, k=10):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_scores)

    # Sort scores and true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = np.array(y_true)[sorted_indices]

    # MRR
    mrr = 0
    for rank, label in enumerate(sorted_labels, start=1):
        if label == 1:
            mrr = 1 / rank
            break

    # nDCG@k
    def dcg(labels, k):
        labels = np.array(labels[:k], dtype=np.float32)
        gains = (2 ** labels - 1) / np.log2(np.arange(2, len(labels) + 2))
        return np.sum(gains)

    ideal_labels = sorted(sorted_labels, reverse=True)
    ndcg_k = dcg(sorted_labels, k) / (dcg(ideal_labels, k) + 1e-10)

    return auc, mrr, ndcg_k

# Plot AUC Curve
def plot_auc_curve(y_true, y_scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend()
    plt.show()

# Main Script
if __name__ == "__main__":
    print("# Load and preprocess data")
    train_news_path = "MINDsmall_train/news.tsv"
    train_behaviors_path = "MINDsmall_train/behaviors.tsv"
    val_news_path = "MINDsmall_dev/news.tsv"
    val_behaviors_path = "MINDsmall_dev/behaviors.tsv"
    test_news_path = "MINDlarge_test/news.tsv"
    test_behaviors_path = "MINDlarge_test/behaviors.tsv"

    print("# Load train, validation and test datasets")
    # Load train and validation datasets
    train_news_df, train_behaviors_df = load_train_data(train_news_path, train_behaviors_path)
    val_news_df, val_behaviors_df = load_validation_data(val_news_path, val_behaviors_path)
    test_news_df, test_behaviors_df = load_test_data(test_news_path, test_behaviors_path)

    print("# Preprocess news")
    train_news_df = preprocess_news(train_news_df)
    val_news_df = preprocess_news(val_news_df)
    test_news_df = preprocess_news(test_news_df)

    print("# Build vocabulary and convert titles to indices")
    word2idx = build_vocab(train_news_df)
    train_news_df = titles_to_indices(train_news_df, word2idx)
    val_news_df = titles_to_indices(val_news_df, word2idx)
    test_news_df = titles_to_indices(test_news_df, word2idx)

    print("# Prepare Training Dataset") 
    #train_titles = torch.tensor(train_news_df['title_indices'].tolist(), dtype=torch.long)
    #train_labels = torch.randint(0, 2, (len(train_titles),), dtype=torch.float)  # Simulated labels for demonstration
    train_titles, train_labels = prepare_train_data(train_behaviors_df, train_news_df)
    train_dataset = NewsDataset(train_titles, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    print("# Prepare Validation Dataset") 
    val_user_impressions, val_labels = prepare_val_data(val_behaviors_df, val_news_df)

    print("# Prepare Testing Dataset") 
    test_user_impressions, test_labels = prepare_test_data(test_behaviors_df, test_news_df)


    print("# Model Initialization") 
    vocab_size = len(word2idx)
    model = NRMS(vocab_size, EMBEDDING_DIM, num_heads=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()


    print("# Training Loop") 
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_auc, val_mrr, val_ndcg_5, val_ndcg_10 = test_model_with_candidates(model, val_user_impressions, val_labels)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}, "
              f"Val AUC: {val_auc:.4f}, MRR: {val_mrr:.4f}, nDCG@5: {val_ndcg_5:.4f}, nDCG@10: {val_ndcg_10:.4f}")

    print("# Testing") 
    test_auc, test_mrr, test_ndcg_5, test_ndcg_10 = test_model_with_candidates(model, test_user_impressions, test_labels)
    print(f"Test Results - AUC: {test_auc:.4f}, MRR: {test_mrr:.4f}, nDCG@5: {test_ndcg_5:.4f}, nDCG@10: {test_ndcg_10:.4f}")
