/*
 * ELP-720 Telecommunication Networks Laboratory - Assignment 2
 * Problem Statement 1: Authenticated LCD Smart Calculator with Keypad and Binary LED Output
 *
 * Author: Sudhanshu Chaudhary (2019JTM2207)
 * Institution: IIT Delhi
 * Simulation Platform: Tinkercad / Arduino Uno
 *
 * Description:
 *  - 4-digit PIN authentication ('1234') with masking ('*').
 *  - 16x2 LiquidCrystal display interfacing.
 *  - 4x4 matrix keypad input for operands and arithmetic operators (+, -, *, /).
 *  - Multi-digit number entry ending with 'c'.
 *  - 4-bit binary LED readout of integer result.
 */

#include <LiquidCrystal.h>
#include <Keypad.h>

// Output pins for 4-bit binary representation of result
const int pin2 = 13;
const int pin3 = 12;
const int pin4 = 11;
const int pin5 = 10;
const int waitTime = 1000;

// LCD pins: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(A0, A1, A2, A3, A4, A5);

const byte ROWS = 4;
const byte COLS = 4;

char hexaKeys[ROWS][COLS] = {
    {'1', '2', '3', '+'},
    {'4', '5', '6', '-'},
    {'7', '8', '9', 'c'},
    {'*', '0', '=', '/'}
};

byte rowPins[ROWS] = { 9, 8, 7, 6 };
byte colPins[COLS] = { 5, 4, 3, 2 };

Keypad customKeypad = Keypad(makeKeymap(hexaKeys), rowPins, colPins, ROWS, COLS);

char keypress() {
    char customKey = customKeypad.getKey();
    while (customKey == 0) {
        customKey = customKeypad.getKey();
    }
    return customKey;
}

void displayBinaryOnLEDs(int val) {
    digitalWrite(pin5, (val & 0x01) ? HIGH : LOW);
    digitalWrite(pin4, (val & 0x02) ? HIGH : LOW);
    digitalWrite(pin3, (val & 0x04) ? HIGH : LOW);
    digitalWrite(pin2, (val & 0x08) ? HIGH : LOW);
    delay(waitTime);
}

void setup() {
    pinMode(pin2, OUTPUT);
    pinMode(pin3, OUTPUT);
    pinMode(pin4, OUTPUT);
    pinMode(pin5, OUTPUT);

    lcd.begin(16, 2);
    lcd.print("----Welcome----");
    lcd.setCursor(0, 1);
    lcd.print("---CALCULATOR---");
    delay(2000);
    lcd.clear();
}

void loop() {
    bool authenticated = false;

    // PIN Authentication Phase
    while (!authenticated) {
        lcd.clear();
        lcd.print("Enter user id:");
        lcd.setCursor(0, 1);

        char a = keypress(); lcd.print('*');
        char b = keypress(); lcd.print('*');
        char c = keypress(); lcd.print('*');
        char d = keypress(); lcd.print('*');

        if (a == '1' && b == '2' && c == '3' && d == '4') {
            lcd.print(" Correct");
            delay(2000);
            authenticated = true;
        } else {
            lcd.print(" Incorrect");
            delay(2000);
        }
    }

    // Input First Operand
    lcd.clear();
    lcd.print("Entr 1st num:");
    lcd.setCursor(0, 1);
    char dig = keypress();
    lcd.write(dig);
    int num1 = (dig - '0');
    while (true) {
        dig = keypress();
        if (dig == 'c') break;
        lcd.write(dig);
        num1 = num1 * 10 + (dig - '0');
    }

    // Input Second Operand
    lcd.clear();
    lcd.print("Entr 2nd num:");
    lcd.setCursor(0, 1);
    dig = keypress();
    lcd.write(dig);
    int num2 = (dig - '0');
    while (true) {
        dig = keypress();
        if (dig == 'c') break;
        lcd.write(dig);
        num2 = num2 * 10 + (dig - '0');
    }

    // Input Operation
    lcd.clear();
    lcd.print("Enter operation:");
    lcd.setCursor(0, 1);
    char op = keypress();
    lcd.write(op);
    delay(1000);

    // Compute Result
    lcd.clear();
    lcd.print("Ans: ");
    int int_ans = 0;
    if (op == '+') {
        int_ans = num1 + num2;
        lcd.print(int_ans);
    } else if (op == '-') {
        int_ans = num1 - num2;
        lcd.print(int_ans);
    } else if (op == '*') {
        int_ans = num1 * num2;
        lcd.print(int_ans);
    } else if (op == '/') {
        if (num2 != 0) {
            float float_ans = (float)num1 / (float)num2;
            int_ans = (int)float_ans;
            lcd.print(float_ans);
        } else {
            lcd.print("DIV BY ZERO");
        }
    }

    // Display binary output on LEDs
    displayBinaryOnLEDs(int_ans);
    delay(3000);
    lcd.clear();
}
