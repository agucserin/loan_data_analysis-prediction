# Peer1: Server role in TCP connection
# Peer1: Having Odd number dataset

import urllib.parse

from local_train_dl import *
from local_train_rf import *
from p2p_comm import *

HOST = '172.30.1.16'
PORT = 8888
BUFFER_SIZE = 4096
SEPARATOR = "--SEP--"
PATH = r"G:\내 드라이브\2024 Summer\code\project_federated\\"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
peer_id = '1'

set_seed(42)
# Hyperparameters for training
rounds = 5
batch_size = 25
learning_rate = 1e-4
epochs = 120

learn = 'd'

df = pd.read_csv(PATH + 'test_set.csv')

X_test = df.drop('loan_status', axis=1).values
y_test = df['loan_status'].values

if learn == 'd':
    test_set = LoanDatasetDL(X_test, y_test)
else:
    test_set = LoanDatasetRF(X_test, y_test)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize criterion and model evaluation lists
criterion = nn.CrossEntropyLoss()
tst_loss_list = []
tst_acc_list = []

# P2P Federated Learning Process
for round in range(1, rounds + 1):
    print(f"\n---------------------- Round {round} Start ----------------------")

    # Local Training
    if learn == 'd':
        local_model, loss_list, acc_list = \
        local_train_dl(path=PATH, peer_id=peer_id, device=device,
        batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=round)

        model_save_path = os.path.abspath(os.path.join(PATH, 'peer1', f'peer1_mdl{round}_2.pt'))
        torch.save(local_model.state_dict(), model_save_path)
    else:
        local_model, loss_list, acc_list = \
        local_train_rf(path=PATH, peer_id=peer_id, device=device,
        batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=round)

        model_save_path = os.path.abspath(os.path.join(PATH, 'peer1', f'peer1_mdl{round}_2.joblib'))
        dump(local_model, model_save_path)    

    # Sending Local Model to Peer2
    input('Send Local trained model to peer2: Press Enter to continue...')
    peer1_send_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, round=round, path=PATH)

    # Receiving Peer2 Model
    input("Receive model from peer2: Press Enter to continue...")
    connected = peer1_recv_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, path=PATH)

    if learn == 'd':
        # Averaging Peer1 & Peer2 models
        if connected:
            new_mdl = avg_mdls(peer_id=peer_id, round=round, path=PATH)
            torch.save(new_mdl.state_dict(), PATH + f"peer1\peer1_mdl{round}_1.pt")

        # Check new model and copy if None
        check_new_mdl(peer_id=peer_id, round=round + 1, path=PATH)

        # Evaluating
        new_model = LoanNet(X_test.shape[1])
        new_model.load_state_dict(torch.load(PATH + f"peer1\peer1_mdl{round}_1.pt"))
        loss, acc = evaluate_dl(device=device, model=new_model, criterion=criterion, test_loader=test_loader)
        print(f'>>> Round {round} - Test Loss: {loss:.4f}  Test Accuracy: {acc:.2f}%')
    else:
        # Averaging Peer1 & Peer2 models
        if connected:
            model_peer1 = load(f'peer1\peer1_mdl{round}_2.joblib')
            model_peer2 = load(f'peer1\\recvd_models\peer2_mdl{round}_2.joblib')

        # 병합된 예측값 계산
        y_pred1 = model_peer1.predict_proba(X_test)
        y_pred2 = model_peer2.predict_proba(X_test)

        # 평균값을 사용하여 최종 예측값 계산
        y_pred_combined = (y_pred1 + y_pred2) / 2
        y_pred_final = np.argmax(y_pred_combined, axis=1)

        # 최종 정확도 계산
        final_accuracy = (accuracy_score(y_test, y_pred_final) * 100)
        print(f'>>> Round {round} - Test Accuracy: {final_accuracy:.2f} %')

    # Plotting Loss and Accuracy curve
    tst_loss_list.append(loss)
    tst_acc_list.append(acc)

    results_dir = os.path.join(PATH, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #Saving Test Loss and Accuracy Data
    np.save(PATH+f'results/test_loss.npy', tst_loss_list)
    np.save(PATH+f'results/test_acc.npy', tst_acc_list)

    # End of round
    print(f"----------------------- Round {round} End ------------------------")

# Plotting, Saving Test Loss and Accuracy
plot_graph(data_name='Test Loss', data=tst_loss_list, cnt=rounds)
plot_graph(data_name='Test Accuracy', data=tst_acc_list, cnt=rounds)

print('\n----------------- End of All Learning Process -----------------\n')
