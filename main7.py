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
    vectorizer = CountVectorizer(max_features=50000, stop_words='english')
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
def preprocess_behaviors(behaviors_df, news_df):
    user_histories, candidate_news, labels = [], [], []
    news_dict = {news_id: vector for news_id, vector in zip(news_df['News_ID'], news_df['Title_Vector'])}

    for _, row in behaviors_df.iterrows():
        history = row['History']
        if pd.isna(history):
            history_vector = []
        else:
            history_vector = [news_dict[news_id] for news_id in history.split() if news_id in news_dict]

        impressions = row['Impressions'].split()
        for impression in impressions:
            news_id, label = impression.split('-')
            if news_id in news_dict:
                candidate_news.append(news_dict[news_id])
                user_histories.append(history_vector)
                labels.append(int(label))

    return user_histories, candidate_news, labels

# Pad sequences
def pad_sequences(data, maxlen):
    return [
        seq[:maxlen] + [[0] * len(seq[0])] * (maxlen - len(seq)) if seq and len(seq) < maxlen else
        [[0] * maxlen] if not seq else seq[:maxlen]
        for seq in data
    ]


# Create Dataset and DataLoader
class MINDDataset(Dataset):
    def __init__(self, user_histories, candidate_news, labels):
        self.user_histories = user_histories
        self.candidate_news = candidate_news
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_histories[idx], self.candidate_news[idx], self.labels[idx]

# Define the NRMS model
class NRMSModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(NRMSModel, self).__init__()
        self.news_encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.user_encoder = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, user_histories, candidate_news):
        # Encode candidate news
        news_representation, _ = self.news_encoder(candidate_news)
        news_attention_weights = torch.softmax(self.attention(news_representation).squeeze(-1), dim=1)
        news_vector = (news_attention_weights.unsqueeze(-1) * news_representation).sum(dim=1)

        # Encode user histories
        user_representation, _ = self.user_encoder(user_histories)
        user_attention_weights = torch.softmax(self.attention(user_representation).squeeze(-1), dim=1)
        user_vector = (user_attention_weights.unsqueeze(-1) * user_representation).sum(dim=1)

        # Compute click probability
        dot_product = torch.sum(user_vector * news_vector, dim=-1)
        prediction = torch.sigmoid(self.fc(dot_product))
        return prediction

# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_histories, candidate_news, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(user_histories, candidate_news).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Evaluate the model
def evaluate_model(model, dev_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for user_histories, candidate_news, labels in dev_loader:
            predictions = model(user_histories, candidate_news).squeeze()
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())
    auc_score = roc_auc_score(all_labels, all_predictions)
    print(f"Validation AUC: {auc_score}")

if __name__ == "__main__":
    print("# Load data")
    # Load data
    news_train_df = load_news_data(news_file)
    news_dev_df = load_news_data(news_file_dev)
    news_train_df, news_dev_df = tokenize_news(news_train_df, news_dev_df)

    behaviors_train_df = load_behaviors_data(behaviors_file)
    behaviors_dev_df = load_behaviors_data(behaviors_file_dev)

    print("# Preprocess data")
    train_user_histories, train_candidate_news, train_labels = preprocess_behaviors(behaviors_train_df, news_train_df)
    dev_user_histories, dev_candidate_news, dev_labels = preprocess_behaviors(behaviors_dev_df, news_dev_df)

    print("# padding data")
    max_history_len = 50
    train_user_histories = pad_sequences(train_user_histories, max_history_len)
    dev_user_histories = pad_sequences(dev_user_histories, max_history_len)


    print("# Tensoring the Data")
    train_user_histories = torch.tensor(train_user_histories, dtype=torch.float32)
    train_candidate_news = torch.tensor(train_candidate_news, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    dev_user_histories = torch.tensor(dev_user_histories, dtype=torch.float32)
    dev_candidate_news = torch.tensor(dev_candidate_news, dtype=torch.float32)
    dev_labels = torch.tensor(dev_labels, dtype=torch.float32)
    

    print("# Creating Class Object")
    train_dataset = MINDDataset(train_user_histories, train_candidate_news, train_labels)
    dev_dataset = MINDDataset(dev_user_histories, dev_candidate_news, dev_labels)

    print("# Creating DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)


    print("# Model Initialization") 
    # Initialize the model
    embedding_dim = 300
    hidden_dim = 128
    nrms_model = NRMSModel(embedding_dim, hidden_dim)


    print("# Defining Loss and Optimized") 
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(nrms_model.parameters(), lr=0.001)


    print("# Training and Evaluating") 
    # Train and evaluate
    train_model(nrms_model, train_loader, criterion, optimizer, epochs=5)
    evaluate_model(nrms_model, dev_loader)
