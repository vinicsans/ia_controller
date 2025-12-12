import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import serial
import time
import os
from collections import deque

# ================= CONFIGURAÇÕES =================
URL_CAMERA = "http://172.16.1.44/capture" 
MODEL_PATH = "modelo-v2.tflite" 
PORTA_SERIAL = "COM5"  # <--- CONFIRA A PORTA NO ARDUINO IDE!
LABELS = ["Circulo", "Triangulo", "Quadrado", "Erro", "Vazio"]

TAMANHO_MEDIA = 10 
CONFIDENCE_THRESHOLD = 0.7 
historico_predicoes = deque(maxlen=TAMANHO_MEDIA)
# =================================================

# --- 1. Inicialização Serial ---
try:
    ser = serial.Serial(PORTA_SERIAL, 115200, timeout=0.1)
    time.sleep(2) # Tempo para o ESP32 reiniciar após abrir a porta
    print(f"Conectado na porta {PORTA_SERIAL}")
except Exception as e:
    print(f"ERRO SERIAL: {e}. Feche o Monitor Serial do Arduino!")
    exit()

# --- 2. Carregar Modelo IA ---
print("Carregando IA...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
h, w, c = input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]

last_sent_label = ""

while True:
    try:
        # 3. Captura da imagem
        img_resp = urllib.request.urlopen(URL_CAMERA, timeout=5)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        if frame is None: continue

        # 4. Processamento P&B
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

            # 7. Envio Serial com Feedback
            if conf >= CONFIDENCE_THRESHOLD:
                if label != last_sent_label:
                    print(f"\n[>>>] ENVIANDO: {label} ({conf*100:.1f}%)")
                    ser.write(f"{label}\n".encode('utf-8')) # Envia com \n
                    ser.flush()
                    
                    # Ouve se o ESP32 respondeu
                    time.sleep(0.1)
                    if ser.in_waiting > 0:
                        echo = ser.readline().decode('utf-8').strip()
                        print(f"[RESPOSTA ESP32]: {echo}")
                    
                    last_sent_label = label

        cv2.imshow("IA System - PC Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"Erro no loop: {e}")
        time.sleep(1)

ser.close()
cv2.destroyAllWindows()