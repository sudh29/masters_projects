/*
 * ELP-720 Telecommunication Networks Laboratory - Assignment 5
 * ESP32 Smart Appliance Energy Monitor & Touch Alert with Blynk IoT
 *
 * Author: Sudhanshu Chaudhary (2019JTM2207)
 * Institution: IIT Delhi
 * Board: ESP32 Dev Module
 *
 * Features:
 *  - Virtual Pins V0, V1 for remote relay/appliance switching (LED, Fan).
 *  - Power consumption calculation (Watts & Energy cost).
 *  - Capacitive touch sensor (T3) threshold alert sent via email notification.
 *  - Enterprise WPA2 WiFi authentication.
 */

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include "esp_wpa2.h"

#define EAP_IDENTITY "jtm192207"
#define EAP_PASSWORD "jtm22072908"

const char* ssid = "IITD_WIFI";
const char* auth = "2PjbSS5zvRf2Bpwu_ccVt9pbur4-DkFf"; // Blynk Auth Token

const int relayPin = 2;
float totalPower = 0;
float totalCost = 0;

float wattLed = 0;
float wattFan = 0;
unsigned long currentTimeLed = 0;
unsigned long currentTimeFan = 0;

int touchThreshold = 40;
volatile bool touchDetected = false;

WidgetLED blynkLed1(V2);
WidgetLED blynkLed2(V3);

void IRAM_ATTR onTouchAlert() {
    touchDetected = true;
}

// Control Virtual Pin V0 (LED / Appliance 1)
BLYNK_WRITE(V0) {
    int pinValue = param.asInt();
    Serial.print("LED State: ");
    Serial.println(pinValue);

    if (pinValue == 1) {
        currentTimeLed = millis();
        blynkLed1.on();
        wattLed = random(900, 1100) / 100.0; // Simulated 9-11 W
        Blynk.virtualWrite(V6, wattLed);
        totalPower += wattLed;
    } else {
        blynkLed1.off();
        unsigned long elapsed = millis() - currentTimeLed;
        float hours = elapsed / 3600000.0;
        float cost = wattLed * hours * 10.0; // Simulated tariff
        totalCost += cost;
        Blynk.virtualWrite(V9, totalCost);
        Blynk.virtualWrite(V8, totalPower);
    }
}

// Control Virtual Pin V1 (Fan / Appliance 2)
BLYNK_WRITE(V1) {
    int pinValue = param.asInt();
    Serial.print("Fan State: ");
    Serial.println(pinValue);

    if (pinValue == 1) {
        currentTimeFan = millis();
        blynkLed2.on();
        wattFan = random(1900, 2100) / 100.0; // Simulated 19-21 W
        Blynk.virtualWrite(V7, wattFan);
        totalPower += wattFan;
    } else {
        blynkLed2.off();
        unsigned long elapsed = millis() - currentTimeFan;
        float hours = elapsed / 3600000.0;
        float cost = wattFan * hours * 10.0;
        totalCost += cost;
        Blynk.virtualWrite(V9, totalCost);
        Blynk.virtualWrite(V8, totalPower);
    }
}

// Threshold Adjustment via Slider
BLYNK_WRITE(V4) {
    touchThreshold = param.asInt();
}

void setup() {
    pinMode(relayPin, OUTPUT);
    digitalWrite(relayPin, LOW);

    Serial.begin(115200);
    delay(100);

    touchAttachInterrupt(T3, onTouchAlert, touchThreshold);

    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);

    esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
    esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
    esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));
    esp_wpa2_config_t config = WPA2_CONFIG_INIT_DEFAULT();
    esp_wifi_sta_wpa2_ent_enable(&config);
    WiFi.begin(ssid);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected, IP: " + WiFi.localIP().toString());

    Blynk.begin(auth, ssid, EAP_PASSWORD);
    Blynk.virtualWrite(V5, "System Active");
}

void loop() {
    Blynk.run();

    if (touchDetected) {
        touchDetected = false;
        Serial.println("[Blynk Alert] Touch threshold crossed!");
        Blynk.email("sudhanshu_alert@example.com", "Security Alert", "Capacitive touch tamper detected on ESP32 node.");
        Blynk.virtualWrite(V5, "Tamper Triggered!");
        delay(3000);
        Blynk.virtualWrite(V5, "System Normal");
    }
}
