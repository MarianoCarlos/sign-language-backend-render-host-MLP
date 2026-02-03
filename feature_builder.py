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

    hand_arrays = []

    for hand in hands:
        pts = np.array(
            [[p["x"], p["y"], p["z"]] for p in hand["points"]],
            dtype=np.float32
        )

        # wrist-relative
        pts -= pts[0]

        # scale normalize (palm size)
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts /= scale

        hand_arrays.append(pts)

    # canonical ordering: leftmost x first
    hand_arrays.sort(key=lambda h: h[0, 0])

    left = hand_arrays[0] if len(hand_arrays) > 0 else np.zeros((21, 3), np.float32)
    right = hand_arrays[1] if len(hand_arrays) > 1 else np.zeros((21, 3), np.float32)

    return np.concatenate([left.flatten(), right.flatten()])
