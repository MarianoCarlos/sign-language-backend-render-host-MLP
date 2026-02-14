import numpy as np

def build_features(hands):
    """
    hands: [
      {
        handedness: "Left" | "Right",
        points: [{x,y,z}, ... 21]
      }
    ]
    returns: np.array shape (126,)
    """

    left = np.zeros((21, 3), np.float32)
    right = np.zeros((21, 3), np.float32)

    for hand in hands:
        pts = np.array(
            [[p["x"], p["y"], p["z"]] for p in hand["points"]],
            dtype=np.float32
        )

        # Wrist-relative normalization
        pts -= pts[0]

        # Scale normalization (palm size)
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts /= scale

        if hand["handedness"] == "Left":
            left = pts
        elif hand["handedness"] == "Right":
            right = pts

    return np.concatenate([left.flatten(), right.flatten()])
