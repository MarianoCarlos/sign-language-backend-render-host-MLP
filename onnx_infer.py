import onnxruntime as ort
import numpy as np
import json

class SignONNX:
    def __init__(self, model_path, labels_path, mean_path, std_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        with open(labels_path) as f:
            self.labels = json.load(f)

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, features):
        x = (features - self.mean) / self.std
        x = x.astype(np.float32)[None, :]

        logits = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )[0]

        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum(axis=1, keepdims=True)

        idx = int(np.argmax(probs))
        return self.labels[str(idx)], float(probs[0, idx])
