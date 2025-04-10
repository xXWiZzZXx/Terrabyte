#include <ModbusMaster.h>

ModbusMaster node;

void setup() {
  Serial.begin(115200);
  Serial2.begin(4800, SERIAL_8N1, 16, 17);  // UART2 sur GPIO 16/17
  node.begin(1, Serial2);  // Adresse esclave 0x01

  Serial.println("Lecture capteur NPK...");
}

void loop() {
  uint8_t result;
  result = node.readInputRegisters(0x0000, 7);

  if (result == node.ku8MBSuccess) {
    float hum  = node.getResponseBuffer(0) / 10.0;
    float temp = node.getResponseBuffer(1) / 10.0;
    int   ec   = node.getResponseBuffer(2);       
    float ph   = node.getResponseBuffer(3) / 10.0;
    int   n    = node.getResponseBuffer(4);
    int   p    = node.getResponseBuffer(5);
    int   k    = node.getResponseBuffer(6);

    Serial.println("Température : " + String(temp) + " °C");
    Serial.println("Humidité : " + String(hum) + " %");
    Serial.println("EC : " + String(ec));
    Serial.println("pH : " + String(ph));
    Serial.println("N : " + String(n));
    Serial.println("P : " + String(p));
    Serial.println("K : " + String(k));
  } else {
    Serial.print("Erreur Modbus : ");
    Serial.println(result);
  }

  delay(3000);
}
