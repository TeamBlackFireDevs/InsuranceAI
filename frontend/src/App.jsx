import React, { useState } from "react";
import { motion } from "framer-motion";
import { Moon, Sun, Maximize } from "lucide-react";

const API_URL = process.env.REACT_APP_API_URL;

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [document, setDocument] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  // toggle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  // handle file upload
  const handleFileUpload = (e) => {
    setDocument(e.target.files[0]);
  };

  // send request to backend
  const handleSubmit = async () => {
    if (!document || !question) {
      alert("Please upload a document and enter a question!");
      return;
    }

    setLoading(true);
    try {
      // step 1: upload file to backend
      const formData = new FormData();
      formData.append("file", document);

      const uploadRes = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const { url } = await uploadRes.json();

      // step 2: send question + file url to API
      const res = await fetch(`${API_URL}/api/v1/hackrx/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer 36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b7993ed5"
        },
        body: JSON.stringify({
          documents: url,
          questions: [question],
        }),
      });

      const data = await res.json();
      setAnswer(data.answers[0]);
    } catch (err) {
      console.error(err);
      setAnswer("Error fetching answer.");
    }
    setLoading(false);
  };

  return (
    <div className={`${darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"} min-h-screen flex flex-col items-center justify-center`}>
      {/* Controls */}
      <div className="absolute top-4 right-4 flex gap-3">
        <button onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? <Sun /> : <Moon />}
        </button>
        <button onClick={toggleFullscreen}><Maximize /></button>
      </div>

      {/* Card */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg w-[500px] flex flex-col gap-4"
      >
        <h1 className="text-2xl font-bold text-center">InsuranceAI</h1>

        <input type="file" accept=".pdf,.docx" onChange={handleFileUpload} className="p-2 border rounded" />

        <textarea
          placeholder="Enter your question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="p-2 border rounded w-full"
        />

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="bg-blue-600 text-white py-2 rounded-xl hover:bg-blue-700"
        >
          {loading ? "Processing..." : "Submit"}
        </button>

        {answer && (
          <div className="p-3 border rounded bg-gray-50 dark:bg-gray-700">
            <strong>Answer:</strong> {answer}
          </div>
        )}
      </motion.div>
    </div>
  );
}
