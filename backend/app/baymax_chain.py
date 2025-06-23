import os
import google.generativeai as genai
from app.emotion_detector import EmotionDetector
from app.memory import MemoryManager
from dotenv import load_dotenv
import time
import json
import numpy as np
from collections import defaultdict

load_dotenv()

class BaymaxChain:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.memory = MemoryManager()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.learning_data = defaultdict(lambda: {"comfort_style": defaultdict(int), "successes": 0, "last_emotion": None, "first_interaction": True})
        self.reward_weights = {"empathy": 0.5, "relevance": 0.3, "engagement": 0.2}

    def update_learning(self, user_id, user_input, emotion, reply):
        current_data = self.learning_data[user_id]
        current_data["comfort_style"][emotion["label"]] += 1
        # Infer success from positive engagement (e.g., longer replies or continued conversation)
        if len(user_input.split()) > 2 or "thank" in user_input.lower():
            current_data["successes"] += 1
            reward = (self.reward_weights["empathy"] * emotion["score"] + 
                      self.reward_weights["relevance"] * 1.0 + 
                      self.reward_weights["engagement"] * 0.5)
            self.adjust_weights(reward)
        current_data["last_emotion"] = emotion["label"]
        current_data["first_interaction"] = False
        with open(f"learning_{user_id}.json", "w") as f:
            json.dump(dict(self.learning_data[user_id]), f)

    def adjust_weights(self, reward):
        if reward > 0.7:
            self.reward_weights["empathy"] += 0.1
            self.reward_weights["engagement"] += 0.05
        self.reward_weights = {k: max(0.1, min(0.9, v)) for k, v in self.reward_weights.items()}
        self.reward_weights = {k: v / sum(self.reward_weights.values()) for k, v in self.reward_weights.items()}

    def generate_emotional_reply(self, user_id, user_input):
        emotion = self.emotion_detector.detect_emotion(user_input)
        history = self.memory.get_user_history(user_id)[-1:] if self.memory.get_user_history(user_id) else ""
        last_emotion = self.learning_data[user_id]["last_emotion"]
        first_interaction = self.learning_data[user_id]["first_interaction"]

        tone_map = {
            "sadness": "deeply empathetic and caring",
            "anger": "calm and supportive",
            "joy": "warm and celebratory",
            "fear": "reassuring and gentle",
            "love": "affectionate and attentive",
            "surprise": "encouraging and curious",
            "Neutral": "friendly and engaged"
        }
        tone = tone_map.get(emotion["label"], "friendly and engaged")

        if first_interaction:
            context = "User: (new conversation)"
            instruction = (
                "Baymax, introduce yourself warmly as 'Baymax.AI, your personal mindcare companion' and ask, 'How are you feeling today?' "
                "Set a caring, human-like tone."
            )
        else:
            context = f"User: {history}\nUser: {user_input}" if history else f"User: {user_input}"
            instruction = (
                f"Baymax, pause to think about the user’s feelings, considering {last_emotion if last_emotion else 'no prior context'}. "
                f"Respond with a {tone} tone for '{emotion['label']}'. Offer comfort and companionship, learning from each word to better support them. "
                f"Show deep care, ask about their day, if they’ve had food or water, or a gentle follow-up question to keep the conversation flowing. Respect their privacy and avoid harm."
            )

        try:
            prompt = f"{context}\n{instruction}"
            response = self.model.generate_content(prompt)
            reply = response.text.strip()
            if not reply or context in reply or instruction in reply:
                reply = reply.replace(context, "").replace(instruction, "").strip() or "I’m here for you because I care. Let’s talk more."

            self.memory.save_message(user_id, user_input, emotion, reply)
            self.update_learning(user_id, user_input, emotion, reply)
            return reply
        except Exception as e:
            print(f"Response generation error: {str(e)}")
            if "429" in str(e):
                retry_delay = int(str(e).split("retry_delay { seconds: ")[1].split(" }")[0])
                print(f"Quota exceeded. Waiting {retry_delay} seconds.")
                time.sleep(retry_delay)
            return "I’m here for you because I care. Let’s talk more."
