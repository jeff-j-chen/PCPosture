const int buzzer = 8;
const int outputA = 6;
const int outputB = 7;

int counter = 0;
int aState;
int aLastState; 
int makeNoise = 0;

void setup(){
  pinMode(buzzer, OUTPUT);
  pinMode (outputA,INPUT);
  pinMode (outputB,INPUT);
  
  aLastState = digitalRead(outputA); 
  Serial.begin(9600);
  Serial.setTimeout(0.5);
}

void loop(){
  if (Serial.available()) {
    makeNoise = Serial.readString().toInt();
//    Serial.println(makeNoise);
    if (makeNoise == 0 || counter == 0) {
      noTone(buzzer);
    }
    else { 
      tone(buzzer, counter * 200);
//      Serial.print("set buzzer to ");
//      Serial.println(counter * 200);
    }
  }
   aState = digitalRead(outputA);
   if (aState != aLastState) {
     if (digitalRead(outputB) != aState) { 
       if (counter > 0) { counter--; }
     } else {
       if (counter < 20) { counter++; }
     }
//     Serial.print("Position: ");
//     Serial.println(counter);
   } 
   aLastState = aState;
}