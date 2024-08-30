# Peer2: Client role in TCP connection
# Peer2: Having Even number dataset

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
peer_id = '2'

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

test_set = LoanDatasetDL(X_test, y_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize criterion and model evaluation lists
criterion = nn.CrossEntropyLoss()
tst_loss_list = []
tst_acc_list = []

###  Main Part  ###
# Receiving Peer1 Model, current round
input("\n\nReceive model from peer1: Press Enter to continue...")
curr_round = peer2_recv_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, path=PATH)

# Check if curr_round is None and handle it
if curr_round is None:
    print("Failed to receive model from peer1.")
else:
    # Local Training
    if learn == 'd':
        local_model, loss_list, acc_list = \
        local_train_dl(path=PATH, peer_id=peer_id, device=device,
        batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=curr_round)

        torch.save(local_model.state_dict(), f'peer2\peer2_mdl{curr_round}_2.pt')
    else:
        local_model, loss_list, acc_list = \
        local_train_rf(path=PATH, peer_id=peer_id, device=device,
        batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=curr_round)

        dump(local_model, f'peer2\peer2_mdl{curr_round}_2.joblib')

    # Sending Local Model to Peer1
    input('Send Local trained model to peer1: Press Enter to continue...')
    peer2_send_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, round=curr_round, path=PATH)

    new_mdl = avg_mdls(peer_id=peer_id, round=curr_round, path=PATH)
    torch.save(new_mdl.state_dict(), f"peer2\peer2_mdl{curr_round}_1.pt")

    # End of round
    print(f"----------------------- Round {curr_round} End ------------------------\n")
