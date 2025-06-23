from tinydb import TinyDB, Query
from datetime import datetime

class MemoryManager:
    def __init__(self):
        self.db = TinyDB("data/user_history.json")

    def get_user_history(self, user_id):
        User = Query()
        history = self.db.search(User.user_id == user_id)
        return [f"User: {entry['message']}\nBaymax: {entry['response']}" for entry in history][-3:]

    def save_message(self, user_id, message, emotion, response):
        self.db.insert({
            "user_id": user_id,
            "message": message,
            "emotion": emotion["label"],
            "response": response,
            "timestamp": datetime.now().isoformat()
        })