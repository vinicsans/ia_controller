import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import serial
import time
import os
from collections import deque

# ================= CONFIGURAÇÕES INICIAIS =================
# Agora o IP é solicitado ao usuário
ip_usuario = input("Digite o IP do ESP32-CAM (ex: 192.168.0.100): ").strip()

# Se o usuário não digitar http://, o código adiciona automaticamente
if not ip_usuario.startswith("http://"):
    URL_CAMERA = f"http://{ip_usuario}/capture"
else:
    URL_CAMERA = f"{ip_usuario}/capture"

print(f"Conectando à câmera em: {URL_CAMERA}")

MODEL_PATH = "modelo-v2.tflite" 
PORTA_SERIAL = "COM5"  # <--- CONFIRA A PORTA!
LABELS = ["Circulo", "Triangulo", "Quadrado", "Erro", "Vazio"]

TAMANHO_MEDIA = 10 
CONFIDENCE_THRESHOLD = 0.7 
historico_predicoes = deque(maxlen=TAMANHO_MEDIA)
# ==========================================================

# --- 1. Inicialização Serial ---
print("Conectando Serial...")
try:
    ser = serial.Serial(PORTA_SERIAL, 115200, timeout=0.1)
    time.sleep(2) # Espera o ESP32 resetar
    print(f"Serial conectado em {PORTA_SERIAL}")
except Exception as e:
    print(f"ERRO SERIAL: {e}")
    print("DICA: Feche o Serial Monitor do Arduino IDE antes de rodar este script!")
    exit()

# --- 2. Carregar Modelo IA ---
print("Carregando IA...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w, c = input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    ser.close()
    exit()

print("\n--- Sistema Pronto! Pressione 'q' na janela da imagem para sair ---\n")

last_sent_label = ""

while True:
    try:
        # 3. Captura da imagem
        img_resp = urllib.request.urlopen(URL_CAMERA, timeout=5)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        
        if frame is None:
            print("Erro: Frame vazio.")
            continue

        # 4. Processamento P&B (Mantendo a lógica original)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(frame_gray, (w, h))
        
        if c == 1:
            img_input = np.expand_dims(img_resized, axis=-1)
        else:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
        input_data = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

        # 5. Inferência
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # 6. Média Móvel (Precisão)
        historico_predicoes.append(output_data)
        
        if len(historico_predicoes) == TAMANHO_MEDIA:
            media = np.mean(historico_predicoes, axis=0)
            idx = np.argmax(media)
            label = LABELS[idx]
            conf = media[idx]

            # Definição de Cores
            if label == "Vazio":
                cor = (200, 200, 200)
            elif label == "Erro":
                cor = (0, 0, 255)
            else:
                cor = (0, 255, 0)

            # 7. Envio Serial
            if conf >= CONFIDENCE_THRESHOLD:
                if label != last_sent_label:
                    print(f"[>>>] ENVIANDO PARA MQTT: {label} ({conf*100:.1f}%)")
                    
                    msg = f"{label}\n"
                    ser.write(msg.encode('utf-8'))
                    ser.flush()
                    
                    last_sent_label = label
                
                texto_display = f"{label}: {conf:.2f}"
            else:
                texto_display = "Processando..."
                cor = (0, 165, 255)

            # Feedback Visual
            frame_display = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame_display, texto_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
            cv2.imshow("IA System - SENAI IoT", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Aguardando conexão ou erro no IP: {e}")
        time.sleep(1)

# Limpeza
try:
    ser.close()
except:
    pass
cv2.destroyAllWindows()
