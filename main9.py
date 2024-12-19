import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Define file paths for MIND small dataset
news_file = "MINDsmall_train/news.tsv"
behaviors_file = "MINDsmall_train/behaviors.tsv"
news_file_dev = "MINDsmall_dev/news.tsv"
behaviors_file_dev = "MINDsmall_dev/behaviors.tsv"


# Load news data
def load_news_data(file_path):
    news_df = pd.read_csv(file_path, sep='\t', header=None, 
                          names=["News_ID", "Category", "Subcategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"])
    return news_df

# Tokenize news titles
def tokenize_news(news_train_df, news_dev_df):
    vectorizer = CountVectorizer(max_features=10000, stop_words='english')  # Reduced max_features to save memory
    news_train_df['Title_Vector'] = list(vectorizer.fit_transform(news_train_df['Title'].fillna(""))
                                         .toarray())
    news_dev_df['Title_Vector'] = list(vectorizer.transform(news_dev_df['Title'].fillna(""))
                                       .toarray())
    return news_train_df, news_dev_df

# Load behavior data
def load_behaviors_data(file_path):
    behaviors_df = pd.read_csv(file_path, sep='\t', header=None, 
                               names=["Impression_ID", "User_ID", "Time", "History", "Impressions"])
    return behaviors_df

# Preprocess user histories and impressions
def preprocess_behaviors(behaviors_df, news_df, sample_fraction=0.1):
    user_histories, candidate_news, labels = [], [], []
    news_dict = {news_id: vector for news_id, vector in zip(news_df['News_ID'], news_df['Title_Vector'])}

    sampled_behaviors_df = behaviors_df.sample(frac=sample_fraction, random_state=42)

    for _, row in sampled_behaviors_df.iterrows():
        impressions = row['Impressions'].split()
        for impression in impressions:
            news_id, label = impression.split('-')
            if news_id in news_dict:
                candidate_news.append(news_dict[news_id])
                user_histories.append([0])  # Placeholder for user history (simplified model)
                labels.append(int(label))

    return user_histories, candidate_news, labels

# Dataset and DataLoader
class MINDDataset(Dataset):
    def __init__(self, user_histories, candidate_news, labels):
        self.user_histories = user_histories
        self.candidate_news = candidate_news
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        candidate_news = torch.tensor(self.candidate_news[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return self.user_histories[idx], candidate_news, label

# Define a simplified neural model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Metrics
def calculate_metrics(y_true, y_pred):
    y_pred = np.array(y_pred)  # Convert to NumPy array
    auc = roc_auc_score(y_true, y_pred)
    sorted_indices = np.argsort(-y_pred)
    y_true_sorted = np.array(y_true)[sorted_indices]

    def dcg_at_k(r, k):
        r = np.asarray(r, dtype=np.float32)[:k]  # Use np.asarray with dtype specified
        return np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(r, k):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return dcg_at_k(r, k) / dcg_max

    nDCG_5 = np.mean([ndcg_at_k(y_true_sorted, 5)])
    nDCG_10 = np.mean([ndcg_at_k(y_true_sorted, 10)])
    MRR = np.mean([1 / (np.where(y_true_sorted == 1)[0][0] + 1)])

    return auc, MRR, nDCG_5, nDCG_10

# Training and Evaluation
def train_model(model, train_loader, criterion, optimizer, dev_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, candidate_news, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(candidate_news).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # Evaluate after each epoch
        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for _, candidate_news, labels in dev_loader:
                predictions = model(candidate_news).squeeze()
                all_labels.extend(labels.numpy())
                all_predictions.extend(predictions.numpy())

        auc, MRR, nDCG_5, nDCG_10 = calculate_metrics(all_labels, all_predictions)
        print(f"Epoch {epoch + 1} - Validation AUC: {auc}")
        print(f"Epoch {epoch + 1} - Validation MRR: {MRR}")
        print(f"Epoch {epoch + 1} - Validation nDCG@5: {nDCG_5}")
        print(f"Epoch {epoch + 1} - Validation nDCG@10: {nDCG_10}")

if __name__ == "__main__":
    # Load and preprocess data
    news_train_df = load_news_data(news_file)
    news_dev_df = load_news_data(news_file_dev)
    news_train_df, news_dev_df = tokenize_news(news_train_df, news_dev_df)

    behaviors_train_df = load_behaviors_data(behaviors_file)
    behaviors_dev_df = load_behaviors_data(behaviors_file_dev)

    train_user_histories, train_candidate_news, train_labels = preprocess_behaviors(behaviors_train_df, news_train_df, sample_fraction=0.1)
    dev_user_histories, dev_candidate_news, dev_labels = preprocess_behaviors(behaviors_dev_df, news_dev_df, sample_fraction=0.1)

    train_dataset = MINDDataset(train_user_histories, train_candidate_news, train_labels)
    dev_dataset = MINDDataset(dev_user_histories, dev_candidate_news, dev_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # Initialize model
    input_dim = len(train_candidate_news[0])  # Dynamically calculate input dimension
    model = SimpleModel(input_dim)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, dev_loader, epochs=5)
