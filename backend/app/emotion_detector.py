from transformers import pipeline
import torch

class EmotionDetector:
    def __init__(self):
        try:
            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier = pipeline(
                "text-classification",
                model="nateraw/bert-base-uncased-emotion",
                device=0 if torch.cuda.is_available() else -1  # 0 for first GPU, -1 for CPU
            )
            print(f"EmotionDetector initialized on {device}")
        except Exception as e:
            raise Exception(f"Failed to load emotion model: {str(e)}")

    def detect_emotion(self, text):
        if not text or not isinstance(text, str):
            return {"label": "Neutral", "score": 0.0}
        try:
            result = self.classifier(text)[0]
            return {"label": result["label"], "score": result["score"]}
        except Exception as e:
            print(f"Emotion detection error: {str(e)}")
            return {"label": "Neutral", "score": 0.0}