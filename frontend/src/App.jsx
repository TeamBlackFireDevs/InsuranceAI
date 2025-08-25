import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Moon, Sun, Maximize, Trash2, X } from "lucide-react";

const API_URL = process.env.REACT_APP_API_URL;

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [questionInput, setQuestionInput] = useState("");
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState([]);
  const [loading, setLoading] = useState(false);

  // darkmode toggle
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  // toggle fullscreen
  const toggleFullscreen = () => {
    if (!window.document.fullscreenElement) {
      window.document.documentElement.requestFullscreen();
    } else {
      window.document.exitFullscreen();
    }
  };

  // handle file upload
  const handleFileUpload = (e) => {
    setUploadedFile(e.target.files[0]);
  };

  // add a question
  const addQuestion = () => {
    if (questionInput.trim()) {
      setQuestions([...questions, questionInput.trim()]);
      setQuestionInput("");
    }
  };

  // remove one question
  const removeQuestion = (index) => {
    setQuestions(questions.filter((_, i) => i !== index));
  };

  // clear all questions
  const clearQuestions = () => {
    setQuestions([]);
    setAnswers([]);
  };

  // send request to backend
  const handleSubmit = async () => {
    if (!uploadedFile || questions.length === 0) {
      alert("Please upload a document and add at least one question!");
      return;
    }

    setLoading(true);
    try {
      // step 1: upload file to backend
      const formData = new FormData();
      formData.append("file", uploadedFile);

      const uploadRes = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const { url } = await uploadRes.json();

      // step 2: send all questions + file url to API
      const res = await fetch(`${API_URL}/api/v1/hackrx/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer 36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b7993ed5",
        },
        body: JSON.stringify({
          documents: url,
          questions: questions,
        }),
      });

      const data = await res.json();
      setAnswers(data.answers || []);
    } catch (err) {
      console.error(err);
      setAnswers(["Error fetching answers."]);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 text-gray-900 dark:bg-neutral-950 dark:text-gray-100">
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
        className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-xl w-[500px] md:w-[800px] min-h-[500px] flex flex-col gap-4"
      >
        <h1 className="text-2xl font-bold text-center">InsuranceAI</h1>

        <input type="file" accept=".pdf,.docx" onChange={handleFileUpload} className="p-2 border rounded mx-auto" />

        {/* Add question */}
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Enter a question..."
            value={questionInput}
            onChange={(e) => setQuestionInput(e.target.value)}
            className="flex-1 p-2 border rounded-xl bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700"
          />
          <button
            onClick={addQuestion}
            className="bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-xl shadow"
          >
            Add
          </button>
        </div>

        {/* List of questions */}
        {questions.length > 0 && (
          <div className="border rounded p-2 flex flex-col gap-2 bg-gray-50 dark:bg-gray-800">
            <div className="flex justify-between items-center">
              <strong>Questions:</strong>
              <button
                onClick={clearQuestions}
                className="text-sm text-red-500 hover:text-red-700 flex items-center gap-1"
              >
                <Trash2 size={16}/> Clear All
              </button>
            </div>
            <ul className="space-y-1">
              {questions.map((q, idx) => (
                <li key={idx} className="flex justify-between items-center bg-white dark:bg-gray-700 p-2 rounded">
                  <span>{q}</span>
                  <button
                    onClick={() => removeQuestion(idx)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <X size={16} />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-xl shadow"
        >
          {loading ? "Processing..." : "Submit"}
        </button>

        {/* Display answers next to questions */}
        {answers.length > 0 && (
          <div className="p-3 border rounded bg-gray-50 dark:bg-gray-700 flex flex-col gap-2">
            <strong>Answers:</strong>
            <ul className="space-y-2">
              {questions.map((q, idx) => (
                <li key={idx} className="bg-white dark:bg-gray-800 p-2 rounded">
                  <p className="font-semibold">Q: {q}</p>
                  <p className="whitespace-pre-line">A: {answers[idx] || "No answer returned."}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </motion.div>
    </div>
  );
}
