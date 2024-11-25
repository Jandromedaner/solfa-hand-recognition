import cv2
import mediapipe as mp
import numpy as np
import pickle

class SolfaGestureRecognizer:
    def __init__(self, model_path="data/solfa_model.pkl"):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.9,
            min_tracking_confidence=0.2
        )
        self.model = self.load_model(model_path)

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


    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            raise

    def predict_gesture(self, landmarks):
        """Predict Solfa gesture from hand landmarks"""
        # Extract features using same method as data collector
        features = self.extract_hand_features(landmarks)
        features = np.array(features).reshape(1, -1)
        return self.model.predict(features)[0]
