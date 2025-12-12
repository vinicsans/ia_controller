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
PORTA_SERIAL = "COM5"  # <--- CONFIRA A PORTA!
LABELS = ["Circulo", "Triangulo", "Quadrado", "Erro", "Vazio"]

TAMANHO_MEDIA = 10 
CONFIDENCE_THRESHOLD = 0.7 
historico_predicoes = deque(maxlen=TAMANHO_MEDIA)
# =================================================

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

print("Sistema Iniciado. Pressione 'q' para sair.")

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

            # Definição de Cores para Desenho
            if label == "Vazio":
                cor = (200, 200, 200) # Cinza/Branco para Vazio
            elif label == "Erro":
                cor = (0, 0, 255)     # Vermelho para Erro
            else:
                cor = (0, 255, 0)     # Verde para Formas

            # 7. Envio Serial (Lógica Principal)
            if conf >= CONFIDENCE_THRESHOLD:
                # Se detectou algo novo (ex: mudou de Quadrado para Vazio)
                if label != last_sent_label:
                    print(f"\n[>>>] ENVIANDO: {label} ({conf*100:.1f}%)")
                    
                    # Envia para o Arduino
                    msg = f"{label}\n"
                    ser.write(msg.encode('utf-8'))
                    ser.flush()
                    
                    last_sent_label = label
                
                # Desenha na tela
                texto_display = f"{label}: {conf:.2f}"
            else:
                # Se a confiança for baixa, mostra "Inseguro"
                texto_display = "Inseguro..."
                cor = (0, 165, 255) # Laranja

            # Feedback Visual na Janela
            # Converte cinza para BGR para poder desenhar colorido
            frame_display = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame_display, texto_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
            cv2.imshow("IA System - PC Vision", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Erro no loop: {e}")
        time.sleep(1)

# Limpeza final
try:
    ser.close()
except:
    pass
cv2.destroyAllWindows()