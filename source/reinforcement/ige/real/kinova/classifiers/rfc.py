"""RandomForest Classifier for collision check"""
import torch
from sklearn.ensemble import RandomForestClassifier

from reinforcement.ige.real.kinova import robot_collision_checker as cc


def train(x_train, y_train):
    # Convert PyTorch tensors to NumPy arrays for sklearn
    rf_x_train = x_train.detach().clone().cpu().numpy()
    rf_y_train = y_train.detach().clone().cpu().numpy().ravel()

    # these hyper-parameters determine model size, subsequent prediction time and impact rest/ros frequencies.
    rf_classifier = RandomForestClassifier(n_estimators=50, min_samples_split=5, min_samples_leaf=1,
                                           max_features='sqrt', random_state=42)
    rf_classifier.fit(rf_x_train, rf_y_train)

    return rf_classifier


def pred(model, x_features):
    device = x_features.device
    assert x_features.dim() == 2, "invalid shape"
    assert x_features.shape[1] == cc.total_feature_cnt, "invalid features"

    rf_x_test = x_features.cpu().numpy()
    pred_probs = model.predict_proba(rf_x_test)[:, 1]
    pred_probs_tensor = torch.tensor(pred_probs, dtype=torch.float32, device=device)
    return pred_probs_tensor.unsqueeze(1)


def pred_labels(model, x_features, prob_threshold=0.5):
    pred_probs = pred(model, x_features)
    return pred_probs, (pred_probs > prob_threshold).float()
