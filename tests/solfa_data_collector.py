import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

class SolfaDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.data = []
        self.labels = []

    def extract_hand_features(self, landmarks):
        """Extract relevant features from hand landmarks"""
        features = []

        # Get all landmark coordinates
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

        # Calculate angles between fingers
        # Thumb
        thumb_angle = self._calculate_finger_angle(
            landmarks.landmark[4],  # Thumb tip
            landmarks.landmark[3],  # Thumb IP
            landmarks.landmark[2]   # Thumb MCP
        )

        # Index
        index_angle = self._calculate_finger_angle(
            landmarks.landmark[8],  # Index tip
            landmarks.landmark[7],  # Index DIP
            landmarks.landmark[6]   # Index PIP
        )

        # Middle
        middle_angle = self._calculate_finger_angle(
            landmarks.landmark[12],
            landmarks.landmark[11],
            landmarks.landmark[10]
        )

        # Ring
        ring_angle = self._calculate_finger_angle(
            landmarks.landmark[16],
            landmarks.landmark[15],
            landmarks.landmark[14]
        )

        # Pinky
        pinky_angle = self._calculate_finger_angle(
            landmarks.landmark[20],
            landmarks.landmark[19],
            landmarks.landmark[18]
        )

        # Add angles to features
        features.extend([thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle])

        # Add distances between fingertips
        fingertips = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = self._calculate_distance(
                    landmarks.landmark[fingertips[i]],
                    landmarks.landmark[fingertips[j]]
                )
                features.append(dist)

        return features

    def _calculate_distance(self, landmark1, landmark2):
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt(
            (landmark1.x - landmark2.x)**2 +
            (landmark1.y - landmark2.y)**2 +
            (landmark1.z - landmark2.z)**2
        )

    def _calculate_finger_angle(self, tip, mid, base):
        """Calculate angle between three points of a finger"""
        vector1 = np.array([tip.x - mid.x, tip.y - mid.y, tip.z - mid.z])
        vector2 = np.array([base.x - mid.x, base.y - mid.y, base.z - mid.z])

        cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    def collect_data(self, gesture_name, num_samples=100):
        """Collect training data for a specific gesture"""
        capture = cv2.VideoCapture(0)
        samples_collected = 0

        while samples_collected < num_samples:
            ret, frame = capture.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]  # Use first hand
                features = self.extract_hand_features(landmarks)
                self.data.append(features)
                self.labels.append(gesture_name)
                samples_collected += 1

            # Display progress
            cv2.putText(frame, f"Collecting {gesture_name}: {samples_collected}/{num_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def save_data(self, filename="solfa_data.pkl"):
        """Save collected data to file"""
        data = {
            'features': self.data,
            'labels': self.labels
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

class SolfaGestureRecognizer:
    def __init__(self, model_path="solfa_model.pkl"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model from file"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict_gesture(self, landmarks):
        """Predict Solfa gesture from hand landmarks"""
        # Extract features using same method as data collector
        features = self.extract_hand_features(landmarks)
        features = np.array(features).reshape(1, -1)
        return self.model.predict(features)[0]

def train_model(data_path="solfa_data.pkl"):
    """Train the gesture recognition model"""
    # Load collected data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = np.array(data['features'])
    y = np.array(data['labels'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("solfa_model.pkl", 'wb') as f:
        pickle.dump(model, f)

    # Print accuracy
    print(f"Model accuracy: {model.score(X_test, y_test):.2f}")
    return model


if __name__ == "__main__":
    collector = SolfaDataCollector()

    # Collect data for each Solfa gesture
    gestures = ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti']
    for gesture in gestures:
        print(f"Collecting data for {gesture}...")
        collector.collect_data(gesture, num_samples=100)

    collector.save_data()

    model = train_model()
