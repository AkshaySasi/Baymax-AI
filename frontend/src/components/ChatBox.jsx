import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";

export default function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [consent, setConsent] = useState(false);
  const messagesEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim() || !consent) return;
    const newMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, newMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await axios.post(`${import.meta.env.VITE_API_URL}/chat`, {
        user_id: "user123", // Replace with dynamic user ID in production
        user_input: input,
      }, {
        headers: { "X-Consent": "true" },
      });
      setMessages((prev) => [...prev, { sender: "ai", text: res.data.response }]);
    } catch (e) {
      setMessages((prev) => [...prev, { sender: "ai", text: "Sorry, I couldn't connect. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="max-w-xl mx-auto p-4 flex-grow flex flex-col">
      {!consent && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-4 p-4 bg-yellow-100 rounded-lg text-gray-800"
        >
          <p className="mb-2">We need your consent to store and process your conversation data to provide a personalized experience.</p>
          <button
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
            onClick={() => setConsent(true)}
          >
            I Consent
          </button>
        </motion.div>
      )}
      <div className="flex-grow h-[60vh] overflow-y-auto border p-4 rounded-xl shadow-md bg-white">
        {messages.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={`mb-3 p-3 rounded-lg max-w-[75%] ${
              msg.sender === "user" ? "bg-blue-100 ml-auto" : "bg-gray-100"
            }`}
          >
            {msg.text}
          </motion.div>
        ))}
        {loading && <p className="text-sm italic text-gray-500">Baymax is thinking...</p>}
        <div ref={messagesEndRef} />
      </div>
      <div className="flex mt-4">
        <input
          className="flex-grow border rounded-l-lg p-2 disabled:bg-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="How are you feeling today?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          disabled={!consent}
        />
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded-r-lg hover:bg-blue-600 transition disabled:bg-gray-400"
          onClick={sendMessage}
          disabled={!consent}
        >
          Send
        </button>
      </div>
    </div>
  );
}