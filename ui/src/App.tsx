import React, { useState, useRef, useEffect } from "react";
import { MdRecycling } from "react-icons/md";

type PredictionResult = {
  prediction: string;
  confidence: number;
};


const App = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<null | PredictionResult>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setResult(null);
    if (!event.target.files || event.target.files.length === 0) {
      setSelectedFile(null);
      return;
    }
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${import.meta.env.BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      console.log(data)
      setResult(data);
    } catch (error) {
      console.error("Failed to fetch prediction:", error);
      alert("An error occurred while classifying the image.");
    } finally {
      setLoading(false);
    }
  };

  console.log(result);

  return (
    <div className="h-screen w-screen p-8 space-y-6 bg-gray-900 flex flex-col items-center">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-50">AI Waste Classifier</h1>
        <p className="text-gray-400">Sort Smarter. Sustain Our Future.</p>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="hidden"
      />

      <div
        className="w-[300px] h-[300px] rounded-full overflow-hidden flex items-center justify-center cursor-pointer border-2 border-dashed border-gray-400 hover:border-cyan-400 transition-colors"
        onClick={() => fileInputRef.current?.click()}
      >
        {preview ? (
          <img
            src={preview}
            alt="Preview"
            className="h-full w-full object-cover"
          />
        ) : (
          <span className="text-gray-400">Click to select image</span>
        )}
      </div>

      {result && (
        <div className="mt-6 text-center p-4 bg-gray-800/50 rounded-lg">
          <p className="text-4xl font-bold text-green-400">{result.prediction.replace("_", " ")}</p>
          <p className="text-md text-gray-400">
            Confidence: {result.confidence}
          </p>
        </div>
      )}

      <MdRecycling size={50} color="#4ade80"/>
      <button
        onClick={handleUpload}
        disabled={!selectedFile}
        className="max-w-[300px] py-3 px-4 rounded-3xl bg-cyan-400 text-transparents font-bold transition duration-200 disabled:bg-gray-500 disabled:cursor-not-allowed hover:enabled:bg-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-opacity-75"
      >
        {loading ? "Classifying..." : "Classify Image"}
      </button>
    </div>
  );
};

export default App;
