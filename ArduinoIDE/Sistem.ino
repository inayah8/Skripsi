// ESP32 + TCS3200 single-read-on-button
// Arduino IDE
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ====== PINS ======
const int S0 = 16;
const int S1 = 17;
const int S2 = 18;
const int S3 = 19;
const int OUT = 34;      // TCS3200 output (pulse)
const int BUTTON_PIN = 4; // tactile switch (active LOW using internal pull-up)
const int LED_PIN = 23;   // white LED for illumination

LiquidCrystal_I2C lcd(0x27, 16, 2); // ganti address jika perlu (0x3F)

// ====== SETTINGS ======
const unsigned long PULSE_TIMEOUT = 2000000UL; // microseconds timeout for pulseIn

// Frequency scaling: S0-S1 = HIGH/LOW => scale x (choose 20% or 100%)
// We'll set S0=HIGH, S1=LOW => S0=1,S1=0 => 20% (safer)
void setScaling(bool s0, bool s1) {
  digitalWrite(S0, s0 ? HIGH : LOW);
  digitalWrite(S1, s1 ? HIGH : LOW);
}

// select color filter for TCS3200: R/G/B
void setFilter(char color) {
  if (color == 'R') { digitalWrite(S2, LOW);  digitalWrite(S3, LOW); } // R
  else if (color == 'G') { digitalWrite(S2, HIGH); digitalWrite(S3, HIGH); } // G
  else if (color == 'B') { digitalWrite(S2, LOW); digitalWrite(S3, HIGH); } // B
  else { digitalWrite(S2, LOW); digitalWrite(S3, LOW); } // default
}

void setup() {
  Serial.begin(115200);
  delay(100);
  pinMode(S0, OUTPUT); pinMode(S1, OUTPUT); pinMode(S2, OUTPUT); pinMode(S3, OUTPUT);
  pinMode(OUT, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP); // button to GND => active LOW
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // set frequency scaling: S0=HIGH, S1=LOW => 20%
  setScaling(true, false);

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Banana Maturity");
  lcd.setCursor(0,1);
  lcd.print("Ready. Press Btn");
}

unsigned long readPulseFreqMicroseconds() {
  // return pulseIn measurement in microseconds for HIGH pulse width
  unsigned long dur = pulseIn(OUT, HIGH, PULSE_TIMEOUT);
  return dur; // if 0 => timeout
}

float freqFromPulseMicros(unsigned long pulseMicros) {
  if (pulseMicros == 0) return 0.0;
  return 1000000.0 / pulseMicros; // approx frequency in Hz
}

// Read R,G,B frequencies and convert to 0..255 values with reference normalization
void readColorRGB(int &R, int &G, int &B) {
  // We will measure multiple times and average to reduce noise
  const int samples = 3;
  float sumR = 0, sumG = 0, sumB = 0;
  for (int s=0; s<samples; s++) {
    setFilter('R');
    delay(70); // allow settling
    unsigned long pR = readPulseFreqMicroseconds();
    float fR = freqFromPulseMicros(pR);

    setFilter('G');
    delay(70);
    unsigned long pG = readPulseFreqMicroseconds();
    float fG = freqFromPulseMicros(pG);

    setFilter('B');
    delay(70);
    unsigned long pB = readPulseFreqMicroseconds();
    float fB = freqFromPulseMicros(pB);

    sumR += fR; sumG += fG; sumB += fB;
    delay(30);
  }
  float aR = sumR / samples;
  float aG = sumG / samples;
  float aB = sumB / samples;

  // Simple normalization: scale relative to the max of (R,G,B) to 0..255
  float maxv = max(aR, max(aG, aB));
  if (maxv <= 0.0) {
    R = G = B = 0;
    return;
  }
  // apply simple ratio scaling
  R = constrain((int)((aR / maxv) * 255.0), 0, 255);
  G = constrain((int)((aG / maxv) * 255.0), 0, 255);
  B = constrain((int)((aB / maxv) * 255.0), 0, 255);
}

// Debounce & wait release
void waitButtonRelease() {
  // Wait until button released (HIGH) with small debounce
  while (digitalRead(BUTTON_PIN) == LOW) delay(10);
  delay(50);
}

void loop() {
  // Poll button: active-low (pressed => LOW)
  if (digitalRead(BUTTON_PIN) == LOW) {
    // short debounce
    delay(20);
    if (digitalRead(BUTTON_PIN) == LOW) {
      // Start single read cycle
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print("Reading...");
      digitalWrite(LED_PIN, HIGH); // turn on illumination

      int R, G, B;
      readColorRGB(R, G, B);

      // show on LCD
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print("R:");
      lcd.print(R);
      lcd.print(" G:");
      lcd.print(G);
      lcd.setCursor(0,1);
      lcd.print("B:");
      lcd.print(B);

      // Send CSV line to Serial: timestamp,R,G,B
      // timestamp as millis()
      unsigned long t = millis();
      Serial.print(t);
      Serial.print(","); Serial.print(R);
      Serial.print(","); Serial.print(G);
      Serial.print(","); Serial.println(B);

      // Visual feedback: blink
      for (int i=0;i<2;i++){
        digitalWrite(LED_PIN, LOW); delay(100);
        digitalWrite(LED_PIN, HIGH); delay(100);
      }
      digitalWrite(LED_PIN, LOW);

      // Wait until button released to avoid multiple reads
      waitButtonRelease();

      // small ready message
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print("Saved & Sent");
      lcd.setCursor(0,1);
      lcd.print("Press Btn again");
    }
  }
  delay(50);
}
