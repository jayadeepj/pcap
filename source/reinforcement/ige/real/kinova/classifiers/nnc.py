"""Neural Network Classifier for collision checking"""
import torch
import torch.nn as nn
import torch.optim as optim

from reinforcement.ige.real.kinova import robot_collision_checker as cc


class KinovaMultiLayerClassifier(nn.Module):

    def __init__(self, input_size):
        super(KinovaMultiLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


def train(model, num_epochs, train_loader, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def pred(model, x_features):
    assert x_features.dim() == 2, "invalid shape"
    assert x_features.shape[1] == cc.total_feature_cnt, "invalid features"

    with torch.no_grad():
        model.eval()
        pred_probs = model(x_features)
        return pred_probs


def pred_labels(model, x_features, prob_threshold=0.5):
    pred_probs = pred(model, x_features)
    return pred_probs, (pred_probs > prob_threshold).float()
