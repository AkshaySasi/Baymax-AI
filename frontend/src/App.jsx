import ChatBox from "./components/ChatBox";

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-100 to-white flex flex-col">
      <h1 className="text-3xl text-center font-bold p-6 text-gray-800">🤖 Baymax.AI</h1>
      <ChatBox />
    </div>
  );
}