#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const uint8_t YAW_CHANNEL = 0;
const uint8_t PITCH_CHANNEL = 1;

const int SERVO_FREQ_HZ = 50;
const int NEUTRAL_US = 1500;
const int MIN_US = 1200;
const int MAX_US = 1800;
const int MAX_DELTA_US = 20;
const unsigned long WATCHDOG_MS = 500;

int yawUs = NEUTRAL_US;
int pitchUs = NEUTRAL_US;
int targetYawUs = NEUTRAL_US;
int targetPitchUs = NEUTRAL_US;
unsigned long lastCommandMs = 0;

int usToTicks(int pulseUs) {
  const float periodUs = 1000000.0 / SERVO_FREQ_HZ;
  return (int)((pulseUs / periodUs) * 4096.0);
}

int clampPulse(int pulseUs) {
  if (pulseUs < MIN_US) return MIN_US;
  if (pulseUs > MAX_US) return MAX_US;
  return pulseUs;
}

int rampToward(int currentUs, int targetUs) {
  targetUs = clampPulse(targetUs);
  if (targetUs > currentUs + MAX_DELTA_US) return currentUs + MAX_DELTA_US;
  if (targetUs < currentUs - MAX_DELTA_US) return currentUs - MAX_DELTA_US;
  return targetUs;
}

void writeServos() {
  pwm.setPWM(YAW_CHANNEL, 0, usToTicks(yawUs));
  pwm.setPWM(PITCH_CHANNEL, 0, usToTicks(pitchUs));
}

int normalizedToPulse(float value) {
  if (value < -1.0) value = -1.0;
  if (value > 1.0) value = 1.0;
  if (value >= 0.0) {
    return NEUTRAL_US + (int)(value * (MAX_US - NEUTRAL_US));
  }
  return NEUTRAL_US + (int)(value * (NEUTRAL_US - MIN_US));
}

void requestNeutral() {
  targetYawUs = NEUTRAL_US;
  targetPitchUs = NEUTRAL_US;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ_HZ);
  delay(10);
  yawUs = NEUTRAL_US;
  pitchUs = NEUTRAL_US;
  targetYawUs = NEUTRAL_US;
  targetPitchUs = NEUTRAL_US;
  writeServos();
  lastCommandMs = millis();
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line == "PING") {
      Serial.println("PONG");
      lastCommandMs = millis();
    } else if (line == "NEUTRAL") {
      requestNeutral();
      lastCommandMs = millis();
      Serial.println("OK NEUTRAL");
    } else if (line.startsWith("SET ")) {
      int firstSpace = line.indexOf(' ');
      int secondSpace = line.indexOf(' ', firstSpace + 1);
      if (secondSpace > 0) {
        float yaw = line.substring(firstSpace + 1, secondSpace).toFloat();
        float pitch = line.substring(secondSpace + 1).toFloat();
        targetYawUs = normalizedToPulse(yaw);
        targetPitchUs = normalizedToPulse(pitch);
        lastCommandMs = millis();
        Serial.println("OK SET");
      } else {
        Serial.println("ERR BAD_SET");
      }
    } else {
      Serial.println("ERR UNKNOWN");
    }
  }

  if (millis() - lastCommandMs > WATCHDOG_MS) {
    requestNeutral();
  }

  yawUs = rampToward(yawUs, targetYawUs);
  pitchUs = rampToward(pitchUs, targetPitchUs);
  writeServos();
  delay(20);
}
