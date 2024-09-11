from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from gtts import gTTS
from googletrans import Translator
import mediapipe as mp
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('action.h5')

# Actions and other configurations
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Initialize Translator
translator = Translator()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(frame):
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get the results
    results = holistic.process(image)
    
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extract face landmarks
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Extract left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extract right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

# Function to detect action and return the recognized word
def recognize_sign(frame):
    global sequence, sentence, predictions
    
    keypoints = extract_keypoints(frame)
    sequence.append(keypoints)
    sequence = sequence[-30:]
    
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))
        
        if np.unique(predictions[-10:])[0] == np.argmax(res): 
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
        
        if len(sentence) > 5:
            sentence = sentence[-5:]
        
        return ' '.join(sentence)
    
    return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Call your sign recognition function
            recognized_sign = recognize_sign(frame)
            
            # Draw the recognized sign on the frame
            cv2.putText(frame, recognized_sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode frame for sending to frontend
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sign_and_audio')
def get_sign_and_audio():
    recognized_sign = ' '.join(sentence)
    target_lang = request.args.get('lang', 'en')
    
    if not recognized_sign:
        return jsonify({'sign': '', 'translation': '', 'audio': ''})
    
    # Translate the recognized sign
    translated_text = translator.translate(recognized_sign, dest=target_lang).text
    
    # Convert to speech using gTTS
    tts = gTTS(text=translated_text, lang=target_lang, slow=False)
    
    # Generate audio without saving
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # Convert audio to base64 to send back to frontend
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    
    return jsonify({
        'sign': recognized_sign,
        'translation': translated_text,
        'audio': audio_base64
    })

if __name__ == '__main__':
    app.run(debug=True)