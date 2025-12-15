#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h> // Biblioteca para MQTT
#include "board_config.h"

// --- CONFIGURAÇÕES WIFI E MQTT ---
const char *ssid = "SENAI IoT";
const char *password = "Senai@IoT";
const char* mqtt_server = "broker.hivemq.com"; // Broker público gratuito
const char* topic_publish = "senai/iot/formas"; // Tópico onde ele enviará os dados

WiFiClient espClient;
PubSubClient client(espClient);

void startCameraServer();

// Função para conectar/reconectar ao Broker MQTT
void reconnect() {
  while (!client.connected()) {
    Serial.print("Tentando conexao MQTT...");
    String clientId = "ESP32CAM-Bridge-" + String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("conectado ao Broker!");
    } else {
      Serial.print("falha, rc=");
      Serial.print(client.state());
      Serial.println(" tentando novamente em 2s");
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(2, OUTPUT); 

  // --- CONFIGURAÇÃO DA CÂMERA ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro na camera: 0x%x", err);
    return;
  }

  // --- CONEXÃO WI-FI ---
  Serial.print("Conectando ao WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.print("."); 
  }
  
  // --- EXIBIÇÃO DO IP E URL ---
  Serial.println("\nWiFi Conectado!");
  Serial.print("Endereço IP: ");
  Serial.println(WiFi.localIP());
  
  Serial.print("URL para o Stream: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/"); // Geralmente o stream inicia na raiz ou /stream
  // ----------------------------

  // --- CONFIGURAÇÃO MQTT ---
  client.setServer(mqtt_server, 1883);

  startCameraServer(); // Inicia stream de vídeo
  Serial.println("SISTEMA_PRONTO");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // ESCUTA A SERIAL (Vindo do Python no USB)
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();

    if (comando.length() > 0) {
      Serial.print("CONFIRMADO_RECEBIMENTO: ");
      Serial.println(comando);

      bool published = client.publish(topic_publish, comando.c_str());
      
      if(published) {
        Serial.println("MQTT: Enviado com sucesso!");
      } else {
        Serial.println("MQTT: Erro ao enviar!");
      }

      digitalWrite(2, HIGH); delay(50); digitalWrite(2, LOW);

      if (comando == "Circulo") {
        // Lógica local aqui
      } 
    }
  }
}
