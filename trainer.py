import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import os
import time

# ================= CONFIGURAÇÕES =================
URL_CAMERA = "http://172.16.1.44/capture" 
MODEL_PATH = "modelo.tflite" 
PASTA_DESTINO = "dataset_treinamento"

if not os.path.exists(PASTA_DESTINO):
    os.makedirs(PASTA_DESTINO)

# --- 1. Carregar Modelo para obter Resolução (w, h) ---
print("Lendo dimensões do modelo TFLite...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    h_ia, w_ia = input_details[0]['shape'][1], input_details[0]['shape'][2]
    print(f"IA configurada para: {w_ia}x{h_ia}")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

print("\n--- COMANDOS ---")
print("Pressione 'S' para iniciar a captura de 100 fotos")
print("Pressione 'Q' para sair")

def capturar_frame():
    """Função auxiliar para buscar a imagem do ESP32"""
    try:
        img_resp = urllib.request.urlopen(URL_CAMERA, timeout=2)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        return cv2.imdecode(imgnp, -1)
    except:
        return None

while True:
    # 2. Mostra o Preview (Ao vivo)
    frame = capturar_frame()
    
    if frame is not None:
        # Criamos uma cópia para mostrar na tela com informações
        preview = frame.copy()
        cv2.putText(preview, "Pressione 'S' para Capturar 100 fotos", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Preview ESP32-CAM", preview)

    key = cv2.waitKey(1) & 0xFF

    # 3. Lógica de Captura ao pressionar 'S'
    if key == ord('s'):
        nome_base = input("\nDigite o nome para o conjunto (ex: Triangulo): ").strip()
        print(f"Iniciando sequência de 100 fotos para '{nome_base}'...")
        
        cont = 1
        while cont <= 100:
            frame_cap = capturar_frame()
            
            if frame_cap is not None:
                # PROCESSAMENTO (Igual à sua IA)
                frame_gray = cv2.cvtColor(frame_cap, cv2.COLOR_BGR2GRAY)
                img_final = cv2.resize(frame_gray, (w_ia, h_ia))

                # SALVAMENTO
                nome_arquivo = f"{nome_base}{cont:03d}.png"
                caminho = os.path.join(PASTA_DESTINO, nome_arquivo)
                cv2.imwrite(caminho, img_final)

                # Feedback Visual na Janela durante a captura
                feedback = frame_cap.copy()
                cv2.putText(feedback, f"CAPTurando: {cont}/100", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Preview ESP32-CAM", feedback)
                cv2.waitKey(1) # Atualiza a janela

                print(f"Salvo {cont}/100", end='\r')
                cont += 1
                time.sleep(0.0002) # Intervalo solicitado

        print(f"\nConcluído! Conjunto '{nome_base}' salvo com sucesso.")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()