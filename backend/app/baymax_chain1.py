import os
import google.generativeai as genai
from app.emotion_detector import EmotionDetector
from app.memory import MemoryManager
from dotenv import load_dotenv
import time

load_dotenv()

class BaymaxChain:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.memory = MemoryManager()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')  

    def generate_emotional_reply(self, user_id, user_input):
        emotion = self.emotion_detector.detect_emotion(user_input)
        history = self.memory.get_user_history(user_id)[-1:] if self.memory.get_user_history(user_id) else ""
        
        tone_map = {
            "sadness": "deeply empathetic",
            "anger": "soothing and calm",
            "joy": "warm and celebratory",
            "fear": "reassuring and gentle",
            "love": "affectionate and supportive",
            "surprise": "encouraging",
            "Neutral": "friendly and attentive"
        }
        tone = tone_map.get(emotion["label"], "friendly and attentive")
        
        context = f"User: {history}\nUser: {user_input}" if history else f"User: {user_input}"
        instruction = f"Baymax, reply with a {tone} tone for '{emotion['label']}'. Be brief, supportive, and consider the conversation."
        
        try:
            prompt = f"{context}\n{instruction}"
            response = self.model.generate_content(prompt)
            reply = response.text.strip()
            if not reply or context in reply or instruction in reply:
                reply = reply.replace(context, "").replace(instruction, "").strip() or "I’m here for you—want to chat?"
            self.memory.save_message(user_id, user_input, emotion, reply)
            return reply
        except Exception as e:
            print(f"Response generation error: {str(e)}")
            if "429" in str(e):
                retry_delay = int(str(e).split("retry_delay { seconds: ")[1].split(" }")[0])
                print(f"Quota exceeded. Waiting {retry_delay} seconds.")
                time.sleep(retry_delay)
            return "I’m here for you. Can you tell me more about how you feel?"