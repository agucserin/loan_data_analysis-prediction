import socket
import threading
import os
import time
import shutil

def check_new_mdl(peer_id, round, path):
    filename = path + f"peer1\peer1_mdl{round}_1.pt"
    
    # 이전 라운드의 모델 파일이 없으면, 복사하여 새 모델 파일을 생성합니다.
    if not os.path.isfile(filename):
        prev_filename = path + f"peer1\peer1_mdl{round-1}_2.pt"
        if os.path.isfile(prev_filename):
            shutil.copy(prev_filename, filename)
        else:
            print(f"이전 라운드의 파일 {prev_filename} 을 찾을 수 없습니다.")

### Sending & Receiving functions for peer1 (TCP server) ###

# Sending current round, peer1 model
def peer1_send_mdl(host, port, bufsize, sep, round, path):
    def send_model():
        try:
            print(f'피어2에게 모델을 전송 중...')
            serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serverSocket.bind((host, port))
            serverSocket.listen(1)
            conn, addr = serverSocket.accept()
            
            filename = path + f'peer1\peer1_mdl{round}_2.pt'
            if not os.path.exists(filename):
                print(f"파일을 찾을 수 없습니다: {filename}")
                conn.close()
                serverSocket.close()
                return

            filesize = os.path.getsize(filename)
            conn.sendall(f"{round}{sep}{filename}{sep}{filesize}".encode())
            time.sleep(1)

            with open(filename, "rb") as f:
                while True:
                    bytes_read = f.read(bufsize)
                    if not bytes_read:
                        break
                    conn.sendall(bytes_read)
            print('피어2에게 모델을 성공적으로 전송함\n\n')
        except Exception as e:
            print(f'모델 전송 중 오류 발생: {e}')
        finally:
            conn.close()
            serverSocket.close()
    
    t = threading.Thread(target=send_model)
    t.start()


# Receiving model from peer2
def peer1_recv_mdl(host, port, bufsize, sep, path):
    connected = False
    try:
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 소켓 재사용 설정
        serverSocket.bind((host, port))
        serverSocket.listen(1)
        conn, addr = serverSocket.accept()
        
        received = conn.recv(bufsize).decode().split(sep)
        curr_round = int(received[0])
        filename = os.path.basename(received[1])  # 파일명만 가져옵니다.
        filesize = int(received[2])
        
        with open(path + r"peer1\recvd_models\\" + filename, "wb") as f:  # 파일을 peer2/local_models/ 경로에 저장합니다.
            total_received = 0
            while total_received < filesize:
                bytes_read = conn.recv(bufsize)
                if not bytes_read:
                    break
                f.write(bytes_read)
                total_received += len(bytes_read)
        
        conn.close()
        serverSocket.close()
        print(f'라운드 {curr_round}에 대해 피어2로부터 모델을 성공적으로 수신함')
        connected = True
    except Exception as e:
        print(f'모델 수신 중 오류 발생: {e}')
    
    return connected


### Sending & Receiving functions for peer2 (TCP client) ###
def peer2_send_mdl(host, port, bufsize, sep, round, path):
    try:
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSocket.connect((host, port))
        filename = path + f'peer2\peer2_mdl{round}_2.pt'
        if not os.path.exists(filename):
            print(f"파일을 찾을 수 없습니다: {filename}")
            clientSocket.close()
            return

        filesize = os.path.getsize(filename)
        clientSocket.sendall(f"{round}{sep}{filename}{sep}{filesize}".encode())
        time.sleep(1)

        with open(filename, "rb") as f:
            while True:
                bytes_read = f.read(bufsize)
                if not bytes_read:
                    break
                clientSocket.sendall(bytes_read)
        print('peer1에게 로컬 모델을 성공적으로 전송함\n\n')
    except ConnectionRefusedError:
        print('연결 거부됨. peer1이 사용 가능한지 확인.')
    except Exception as e:
        print(f'모델 전송 중 오류 발생: {e}')
    finally:
        clientSocket.close()
        
# Receiving current round and peer1 model
def peer2_recv_mdl(host, port, bufsize, sep, path):
    curr_round = None
    try:
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSocket.connect((host, port))
        received = clientSocket.recv(bufsize).decode().split(sep)
        curr_round = int(received[0])
        filename = os.path.basename(received[1])
        filesize = int(received[2])

        open_path = os.path.join(path, "peer2\\recvd_models\\", filename)

        with open(open_path, "wb") as f:
            total_received = 0
            while total_received < filesize:
                bytes_read = clientSocket.recv(bufsize)
                if not bytes_read:
                    break
                f.write(bytes_read)
                total_received += len(bytes_read)
        print(f'라운드 {curr_round}에 대해 peer1로부터 모델을 성공적으로 수신함')
    except ConnectionRefusedError:
        print('연결 거부됨. peer1이 사용 가능하고 수신 대기 중인지 확인하십시오.')
    except Exception as e:
        print(f'모델 수신 중 오류 발생: {e}')
    finally:
        clientSocket.close()
    return curr_round
