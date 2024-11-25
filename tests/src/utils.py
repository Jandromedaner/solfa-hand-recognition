import numpy as np

def calculate_finger_angle(tip, mid, base):
    """Calculate angle between three points of a finger"""
    vector1 = np.array([tip.x - mid.x, tip.y - mid.y, tip.z - mid.z])
    vector2 = np.array([base.x - mid.x, base.y - mid.y, base.z - mid.z])

    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    return np.sqrt(
        (landmark1.x - landmark2.x)**2 +
        (landmark1.y - landmark2.y)**2 +
        (landmark1.z - landmark2.z)**2
    )
