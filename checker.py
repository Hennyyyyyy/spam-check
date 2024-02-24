import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel('WELOVEDATA.xlsx')

X = df[['title', 'text']]  # Features
y = df['type']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SpamDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpamDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, X, y, vectorizer):
        self.X = X
        self.y = y
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_text = self.X.iloc[idx]['title'] + ' ' + self.X.iloc[idx]['text']
        x_vector = self.vectorizer.transform([x_text]).toarray().flatten()
        y_label = 1 if self.y.iloc[idx] == 'spam' else 0
        return x_vector, y_label

vectorizer = CountVectorizer()
X_train['title'].fillna('', inplace=True)
X_train['text'].fillna('', inplace=True)
vectorizer.fit(X_train['title'] + ' ' + X_train['text'])
train_dataset = CustomDataset(X_train, y_train, vectorizer)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(X_test, y_test, vectorizer)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

print(f"Input Size: {len(vectorizer.get_feature_names_out())}")

input_size = len(vectorizer.get_feature_names_out())
hidden_size = 128
num_classes = 2  # Binary classification

model = SpamDetector(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.float(), labels.long()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.float(), labels.long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {(100 * correct / total):.2f}%')


torch.save(model.state_dict(), 'spam_model.pth')
print('Model saved successfully!')

model = SpamDetector(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('spam_model.pth'))
model.eval()

def predict_spam(model, vectorizer, title, email):
    x_text = title + ' ' + email
    x_vector = vectorizer.transform([x_text]).toarray().flatten()
    with torch.no_grad():
        inputs = torch.tensor(x_vector).float().unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

title = 'You one a FREE cruise!!!!!!'
email = """Congratulations lucky winner, you won a FREE CRUISE! There are no strings attached on this delightful journey. We will visit many beautiful places, and you will meet many nce people! Click here to recieve your free cruise today! https://bit.ly/free-cruise"""

prediction = predict_spam(model, vectorizer, title, email)
print('Spam' if prediction == 1 else 'Not Spam')