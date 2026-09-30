/*
 * ELP-720 Telecommunication Networks Laboratory - Assignment 4
 * ESP32 WiFi Weather Station with Telegram Bot & Capacitive Touch Security Alert
 *
 * Author: Sudhanshu Chaudhary (2019JTM2207)
 * Institution: IIT Delhi
 * Board: ESP32 Dev Module
 *
 * Peripherals:
 *  - 16x2 LCD (pins: 22, 23, 5, 18, 19, 21)
 *  - Status LEDs: Pin 4, 25, 32 (Alert LED), 33
 *  - Capacitive Touch Pin: T3 (GPIO 15)
 *  - WiFi: IITD Enterprise / WPA2 WiFi
 *  - Cloud Services: Telegram Bot API, OpenWeatherMap REST API
 */

#include <LiquidCrystal.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <UniversalTelegramBot.h>
#include <HTTPClient.h>
#include "esp_wpa2.h"

// LCD Configuration: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(22, 23, 5, 18, 19, 21);

// Institute WPA2 Enterprise Credentials (placeholders)
#define EAP_IDENTITY "jtm192207"
#define EAP_PASSWORD "jtm22072908"

const char* ssid = "IITD_WIFI";
const char* weatherApiKey = "da76b2b69eb1d79fa2cd4d90"; // OpenWeatherMap API Key

// Telegram BOT Token
#define BOT_TOKEN "1003941905:AAHSLltbvQjrSyjVqDTfAx8AY-Cc_q98sUs"

WiFiClientSecure client;
UniversalTelegramBot bot(BOT_TOKEN, client);

const int botScanInterval = 1000;
long botLastTime = 0;

const int touchThreshold = 40;
volatile bool touchDetected = false;

void IRAM_ATTR onTouchAlert() {
    touchDetected = true;
}

void handleNewMessages(int numNewMessages) {
    for (int i = 0; i < numNewMessages; i++) {
        String chatId = String(bot.messages[i].chat_id);
        String city = bot.messages[i].text;

        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            String url = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&APPID=" + weatherApiKey;
            http.begin(url);
            int httpCode = http.GET();

            if (httpCode > 0) {
                String payload = http.getString();

                // Parse temperature (Kelvin to Celsius)
                int tempIdx = payload.indexOf("\"temp\":");
                if (tempIdx != -1) {
                    int endTemp = payload.indexOf(",", tempIdx);
                    String tempStr = payload.substring(tempIdx + 7, endTemp);
                    float kelvin = tempStr.toFloat();
                    int celsius = (int)(kelvin - 273.15);

                    // Parse humidity
                    int humIdx = payload.indexOf("\"humidity\":");
                    String humStr = "N/A";
                    if (humIdx != -1) {
                        int endHum = payload.indexOf(",", humIdx);
                        humStr = payload.substring(humIdx + 11, endHum);
                    }

                    // Parse pressure
                    int pressIdx = payload.indexOf("\"pressure\":");
                    String pressStr = "N/A";
                    if (pressIdx != -1) {
                        int endPress = payload.indexOf(",", pressIdx);
                        pressStr = payload.substring(pressIdx + 11, endPress);
                    }

                    // Send Telegram Response
                    String response = "Weather for " + city + ":\n" +
                                     "Temp: " + String(celsius) + " C\n" +
                                     "Humidity: " + humStr + "%\n" +
                                     "Pressure: " + pressStr + " hPa";
                    bot.sendMessage(chatId, response, "");

                    // Update LCD
                    lcd.clear();
                    lcd.setCursor(0, 0);
                    lcd.print(city + " " + String(celsius) + "C");
                    lcd.setCursor(0, 1);
                    lcd.print("Hum:" + humStr + "% P:" + pressStr);
                }
            } else {
                bot.sendMessage(chatId, "Error fetching weather data for " + city, "");
            }
            http.end();
        }
    }
}

void setup() {
    Serial.begin(115200);

    lcd.begin(16, 2);
    lcd.clear();
    lcd.print("----WELCOME----");
    lcd.setCursor(0, 1);
    lcd.print("-WEATHER STATION-");
    delay(1500);

    pinMode(4, OUTPUT);
    pinMode(25, OUTPUT);
    pinMode(32, OUTPUT); // Security Alert LED
    pinMode(33, OUTPUT);

    // Attach capacitive touch interrupt on T3 (GPIO 15)
    touchAttachInterrupt(T3, onTouchAlert, touchThreshold);

    lcd.clear();
    lcd.print("Connecting WiFi");
    WiFi.mode(WIFI_STA);

    // Setup WPA2 Enterprise if connecting to campus network
    esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
    esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
    esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));
    esp_wpa2_config_t config = WPA2_CONFIG_INIT_DEFAULT();
    esp_wifi_sta_wpa2_ent_enable(&config);
    WiFi.begin(ssid);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected, IP: " + WiFi.localIP().toString());
        lcd.clear();
        lcd.print("WiFi Connected");
        lcd.setCursor(0, 1);
        lcd.print(WiFi.localIP().toString());
    } else {
        Serial.println("\nWiFi connection timed out");
        lcd.clear();
        lcd.print("WiFi Offline");
    }
    delay(2000);

    lcd.clear();
    lcd.print("Send City to");
    lcd.setCursor(0, 1);
    lcd.print("Telegram Bot");
}

void loop() {
    // Check for incoming Telegram queries
    if (millis() > botLastTime + botScanInterval) {
        int numNewMessages = bot.getUpdates(bot.last_message_received + 1);
        if (numNewMessages > 0) {
            digitalWrite(33, HIGH);
            handleNewMessages(numNewMessages);
            digitalWrite(33, LOW);
        }
        botLastTime = millis();
    }

    // Handle touch tamper detection
    if (touchDetected) {
        touchDetected = false;
        Serial.println("[ALERT] Capacitive Touch Tamper Detected!");
        digitalWrite(32, HIGH);
        delay(3000);
        digitalWrite(32, LOW);
    }
}
