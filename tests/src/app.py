import cv2
import pygame
import numpy as np
from .gesture_recognizer import SolfaGestureRecognizer
from enum import Enum
import time

class SolfaGesture(Enum):
    DO = "do"
    DI = "di"
    RA = "ra"
    RE = "re"
    RI = "ri"
    ME = "me"
    MI = "mi"
    FA = "fa"
    FI = "fi"
    SE = "se"
    SO = "so"
    SI = "si"
    LE = "le"
    LA = "la"
    LI = "li"
    TE = "te"
    TI = "ti"

class SolfaLearningApp:
    def __init__(self):
        self.recognizer = SolfaGestureRecognizer()
        self.current_target = None
        self.score = 0
        self.gesture_sequence = list(SolfaGesture)
        self.current_gesture_index = 0
        self.correct_gesture_start_time = None

        # Initialize pygame for sound
        pygame.init()
        pygame.mixer.init()

        # Load sounds (you'll need to add these sound files)
        self.success_sound = pygame.mixer.Sound('data/sounds/success.wav')
        self.failure_sound = pygame.mixer.Sound('data/sounds/failure.wav')

        # Drawing properties
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2

        # Initialize target gesture
        self.set_next_target()

    def set_next_target(self):
        """Set the next gesture in sequence"""
        self.current_target = self.gesture_sequence[self.current_gesture_index].value
        self.current_gesture_index = (self.current_gesture_index + 1) % len(self.gesture_sequence)

    def draw_interface(self, frame, detected_gesture=None):
        """Draw the user interface on the frame"""
        # Draw target gesture
        cv2.putText(frame, f"Target: {self.current_target}",
                   (10, 30), self.font, self.font_scale, (0, 255, 0), self.thickness)

        # Draw score
        cv2.putText(frame, f"Score: {self.score}",
                   (10, 70), self.font, self.font_scale, (0, 255, 0), self.thickness)

        # Draw detected gesture if any
        if detected_gesture:
            cv2.putText(frame, f"Detected: {detected_gesture}",
                      (10, 110), self.font, self.font_scale, (255, 0, 0), self.thickness)

        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 'n' for next gesture",
                   (10, frame.shape[0] - 20), self.font, 0.6, (255, 255, 255), 1)

    def provide_feedback(self, frame, is_correct):
        """Provide visual and audio feedback"""
        if is_correct:
            # Draw green check mark
            cv2.putText(frame, "✓", (frame.shape[1]//2, frame.shape[0]//2),
                       self.font, 3, (0, 255, 0), 3)
            self.success_sound.play()
        else:
            # Draw red X
            cv2.putText(frame, "✗", (frame.shape[1]//2, frame.shape[0]//2),
                       self.font, 3, (0, 0, 255), 3)
            self.failure_sound.play()

    def run(self):
        """Main application loop"""
        capture = cv2.VideoCapture(0)
        previous_time = 0
        showing_feedback = False

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                continue

            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
#            frame = cv2.resize(frame, (800, 600))

            # Process frame with gesture recognizer
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.recognizer.hands.process(frame_rgb)

            # Draw the basic interface
            self.draw_interface(frame)

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                # Draw hand landmarks
                self.recognizer.mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    self.recognizer.mp_hands.HAND_CONNECTIONS
                )

                # Get gesture prediction
                predicted_gesture = self.recognizer.predict_gesture(results.multi_hand_landmarks[0])
                self.draw_interface(frame, predicted_gesture)

                # Check if gesture matches target
                if predicted_gesture == self.current_target and not showing_feedback:
                    if self.correct_gesture_start_time is None:
                        self.correct_gesture_start_time = time.time()  # Start tracking time
                    else:
                        elapsed_time = time.time() - self.correct_gesture_start_time
                        if elapsed_time >= 0.5:
                            self.score += 1
                            self.provide_feedback(frame, True)
                            showing_feedback = True
                            feedback_timer = cv2.getTickCount()
                            self.correct_gesture_start_time = None  # Reset timer

            # Handle feedback timer
            if showing_feedback:
                if (cv2.getTickCount() - feedback_timer) / cv2.getTickFrequency() > 1.0:
                    showing_feedback = False
                    self.set_next_target()

            # Display the frame
            cv2.imshow('Solfa Learning App', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.set_next_target()
                showing_feedback = False

        # Cleanup
        capture.release()
        cv2.destroyAllWindows()
        pygame.quit()
