import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # 이 부분 추가
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class LoanDatasetDL(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
"""
#512
class LoanNet(nn.Module):
    def __init__(self, input_dim):
        super(LoanNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)  # 드롭아웃 비율 변경
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.batch_norm5 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm5(self.fc5(x)))
        x = self.fc6(x)
        return x
"""
#1024

class LoanNet(nn.Module):
    def __init__(self, input_dim):
        super(LoanNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.35)  # 드롭아웃 비율 변경
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)
        self.batch_norm6 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm5(self.fc5(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm6(self.fc6(x)))
        x = self.fc7(x)
        return x


def local_train_dl(path=None, peer_id=None, device=None, batch_size=16, learning_rate=0.001, epochs=5, curr_round=None):
    set_seed(42)
    # 데이터 로드 및 전처리
    if peer_id == '1':
        df = pd.read_csv(path + 'train_set_1.csv')
    else:
        df = pd.read_csv(path + 'train_set_2.csv')

    # 특성과 타겟 분리
    X = df.drop('loan_status', axis=1).values
    y = df['loan_status'].values

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 데이터셋 및 데이터로더 구성
    dataset = LoanDatasetDL(X, y)
    g = torch.Generator().manual_seed(42)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=0, worker_init_fn=lambda _: set_seed(42))

    # 모델 초기화
    model = LoanNet(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # 이전 라운드 모델 파라미터 로드
    if curr_round > 1:
        model_path = path + f'peer{peer_id}\peer{peer_id}_mdl{curr_round-1}_1.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

    # 훈련 과정
    model.train()
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)


            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step(epoch_loss)

        epoch_loss /= len(data_loader)
        epoch_acc = 100 * correct / total
        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return model, loss_list, acc_list

def avg_mdls(peer_id, round, path):
    # Averaging Peer1 & Peer2 models
    if peer_id == '1':
        peer1_model_path = path + f'peer1\peer1_mdl{round}_2.pt'
        peer2_model_path = path + f'peer1\\recvd_models\peer2_mdl{round}_2.pt'
    else:
        peer1_model_path = path + f'peer2\peer2_mdl{round}_2.pt'
        peer2_model_path = path + f'peer2\\recvd_models\peer1_mdl{round}_2.pt'

    peer1_model = torch.load(peer1_model_path)
    peer2_model = torch.load(peer2_model_path)

    new_model = LoanNet(input_dim=peer1_model['fc1.weight'].shape[1])  # Ensure correct input dimension
    new_state_dict = avg_dicts([peer1_model, peer2_model], 2)
    new_model.load_state_dict(new_state_dict)

    return new_model

def avg_dicts(dicts, n_clnt):
    # Averaging model state dictionaries
    avg_dict = dicts[0]
    for key in avg_dict.keys():
        for i in range(1, n_clnt):
            avg_dict[key] += dicts[i][key]
        avg_dict[key] = torch.div(avg_dict[key], n_clnt)
    return avg_dict

def evaluate_dl(device, model, criterion, test_loader):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(test_loader.dataset)
    tst_acc = 100 * correct / total
    return loss, tst_acc

def plot_graph(data_name, data, cnt):
    plt.plot(range(1, cnt+1), data)
    plt.title(f'{data_name}')
    plt.xlabel('Round')
    plt.ylabel(f'{data_name}')
    plt.show()
