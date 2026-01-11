
import React, { useState, useRef } from "react";

export default function UploadOMR() {
  const [files, setFiles] = useState([]);
  const [templateFile, setTemplateFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState(null);
  const [results, setResults] = useState(null);
  const [answerKey, setAnswerKey] = useState("");
  const inputRef = useRef(null);
  const templateRef = useRef(null);

  const handleFiles = (fileList) => {
    const arr = Array.from(fileList);
    const allowed = arr.filter((f) => f.size <= 10 * 1024 * 1024);
    setFiles((prev) => [...prev, ...allowed]);
  };

  const onSelectFiles = (e) => handleFiles(e.target.files);
  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const removeFile = (index) =>
    setFiles((prev) => prev.filter((_, i) => i !== index));

  const isValidJSON = (str) => {
    try {
      JSON.parse(str);
      return true;
    } catch {
      return false;
    }
  };

  const handleTemplateSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setTemplateFile(e.target.files[0]);
    }
  };

  const uploadFiles = async () => {
    if (!files.length) return;

    // Validate JSON
    if (answerKey.trim() && !isValidJSON(answerKey)) {
      setMessage({ type: "error", text: "❌ Invalid JSON Answer Key" });
      return;
    }

    setUploading(true);
    setMessage(null);
    setProgress(0);
    setResults(null);

    const formData = new FormData();

    // 🔥 SEND ANSWER KEY FIRST
    formData.append("answer_key", answerKey || "{}");

    // 🔥 SEND TEMPLATE JSON IF SELECTED
    if (templateFile) {
      // We will read the file and send its content as a string, 
      // or just send the file and let backend read it.
      // The backend expects "template_json" as a string in Form(None) usually, 
      // or we can send it as a file.
      // Let's read it here to be safe and consistent with the user's request "send template ka json".
      // Actually sending as a file is often cleaner for large JSONs, but let's see.
      // The user said "template ka json".
      // Let's send it as a file with key 'template_file' or string 'template_json'.
      // Backend changes will match this. I'll read it as text here.
      const text = await templateFile.text();
      if (!isValidJSON(text)) {
        setMessage({ type: "error", text: "❌ Invalid Template JSON File" });
        setUploading(false);
        return;
      }
      formData.append("template_json", text);
    }

    // 🔥 SEND FILES
    files.forEach((f) => formData.append("files", f));

    // ---- XHR upload (FastAPI-friendly) ----
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:8000/process-omr");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable)
        setProgress(Math.round((e.loaded / e.total) * 100));
    };

    xhr.onload = () => {
      setUploading(false);

      if (xhr.status >= 200 && xhr.status < 300) {
        const res = JSON.parse(xhr.responseText);
        setMessage({ type: "success", text: "✅ OMR Processed Successfully!" });
        setResults(res);
        setFiles([]);
        setAnswerKey("");
        setTemplateFile(null); // Clear template
      } else {
        setMessage({
          type: "error",
          text: `❌ Upload failed: ${xhr.statusText || xhr.status}`,
        });
      }
    };

    xhr.onerror = () => {
      setUploading(false);
      setMessage({ type: "error", text: "❌ Network Error" });
    };

    xhr.send(formData);
  };

  return (
    <div className="min-h-screen flex flex-col bg-white p-6">
      <h3 className="text-2xl font-bold mb-6">Upload & Process OMR Sheets</h3>

      <div
        className="border-2 border-dashed p-6 rounded-xl flex flex-col md:flex-row"
        onDrop={onDrop}
        onDragOver={onDragOver}
      >
        {/* LEFT SIDE */}
        <div className="md:w-2/3">
          <p className="text-gray-700 mb-4">
            Drag & drop scanned OMR sheets here, or
          </p>

          <div className="flex items-center gap-4 mb-4">
            <input
              ref={inputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={onSelectFiles}
              className="hidden"
            />

            <button
              onClick={() => inputRef.current.click()}
              className="px-4 py-2 bg-blue-500 text-white rounded-full"
            >
              Choose files
            </button>

            <button
              onClick={() => setFiles([])}
              disabled={!files.length}
              className="px-4 py-2 bg-gray-200 rounded-full disabled:opacity-50"
            >
              Clear
            </button>

            <button
              onClick={uploadFiles}
              disabled={!files.length || uploading}
              className="px-4 py-2 bg-green-500 text-white rounded-full disabled:opacity-50"
            >
              {uploading ? "Uploading..." : "Send OMR Sheets"}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* ANSWER KEY INPUT */}
            <div>
              <label className="block text-gray-700 font-medium mb-1">
                Custom Answer Key (JSON)
              </label>
              <textarea
                value={answerKey}
                onChange={(e) => setAnswerKey(e.target.value)}
                placeholder='{"1":3,"2":1,...}'
                className="w-full p-2 border rounded-md h-32 text-sm font-mono"
              />
            </div>

            {/* TEMPLATE UPLOAD */}
            <div>
              <label className="block text-gray-700 font-medium mb-1">
                Template JSON (Optional)
              </label>
              <div className="border p-4 rounded-md h-32 flex flex-col justify-center items-center text-center bg-gray-50">
                <input
                  type="file"
                  accept=".json"
                  ref={templateRef}
                  onChange={handleTemplateSelect}
                  className="hidden"
                />
                <button
                  onClick={() => templateRef.current.click()}
                  className="text-blue-600 text-sm font-semibold hover:underline"
                >
                  {templateFile ? templateFile.name : "+ Select Template JSON"}
                </button>
                {templateFile && (
                  <button
                    onClick={() => setTemplateFile(null)}
                    className="text-red-500 text-xs mt-2"
                  >
                    Remove
                  </button>
                )}
                <p className="text-xs text-gray-400 mt-2">
                  If not provided, default template will be used.
                </p>
              </div>
            </div>
          </div>

          {message && (
            <div
              className={`p-3 mt-3 rounded ${message.type === "success"
                  ? "bg-green-100 text-green-700"
                  : "bg-red-100 text-red-700"
                }`}
            >
              {message.text}
            </div>
          )}

          {uploading && (
            <div className="mt-3">
              <div className="w-full bg-gray-200 h-3 rounded">
                <div
                  className="h-3 bg-green-500 rounded"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="text-sm mt-1">{progress}%</p>
            </div>
          )}

          {results && (
            <div className="mt-6 p-4 bg-gray-100 rounded-lg max-h-64 overflow-auto">
              <h4 className="font-semibold mb-2">Results:</h4>
              <pre className="text-sm">{JSON.stringify(results, null, 2)}</pre>
            </div>
          )}
        </div>

        {/* RIGHT SIDE FILE PREVIEW */}
        <div className="md:w-1/3 mt-6 md:mt-0 md:pl-6">
          <div className="bg-gray-50 p-4 rounded-lg max-h-56 overflow-auto">
            {files.length === 0 ? (
              <p className="text-gray-500">No files selected</p>
            ) : (
              files.map((f, i) => (
                <div
                  key={i}
                  className="flex justify-between py-2 border-b last:border-none"
                >
                  <div>
                    <p className="text-sm">{f.name}</p>
                    <p className="text-xs text-gray-500">
                      {(f.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                  <button
                    onClick={() => removeFile(i)}
                    className="text-red-500 text-sm"
                  >
                    Remove
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


// import React, { useState, useRef } from "react";

// export default function UploadOMR() {
//   const [files, setFiles] = useState([]);
//   const [uploading, setUploading] = useState(false);
//   const [progress, setProgress] = useState(0);
//   const [message, setMessage] = useState(null);
//   const [results, setResults] = useState(null);
//   const [answerKey, setAnswerKey] = useState("");
//   const inputRef = useRef(null);

//   const handleFiles = (fileList) => {
//     const arr = Array.from(fileList);
//     const allowed = arr.filter((f) => f.size <= 10 * 1024 * 1024);
//     setFiles((prev) => [...prev, ...allowed]);
//   };

//   const onSelectFiles = (e) => handleFiles(e.target.files);
//   const onDrop = (e) => {
//     e.preventDefault();
//     if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
//   };

//   const removeFile = (index) =>
//     setFiles((prev) => prev.filter((_, i) => i !== index));

//   const isValidJSON = (str) => {
//     try {
//       JSON.parse(str);
//       return true;
//     } catch {
//       return false;
//     }
//   };

//   const uploadFiles = () => {
//     if (!files.length) return;

//     if (answerKey.trim() && !isValidJSON(answerKey)) {
//       setMessage({ type: "error", text: "❌ Invalid JSON Answer Key" });
//       return;
//     }

//     setUploading(true);
//     setMessage(null);
//     setProgress(0);
//     setResults(null);

//     const formData = new FormData();
//     formData.append("answer_key", answerKey || "{}");

//     files.forEach((file) => formData.append("files", file));

//     const xhr = new XMLHttpRequest();
//     xhr.open("POST", "http://localhost:8000/process-omr", true);

//     // 🔥 CORS FIX
//     xhr.withCredentials = false;
//     xhr.setRequestHeader("Accept", "application/json");

//     // Upload progress
//     xhr.upload.onprogress = (e) => {
//       if (e.lengthComputable) {
//         const percent = Math.round((e.loaded / e.total) * 100);
//         setProgress(percent);
//       }
//     };

//     xhr.onload = () => {
//       setUploading(false);

//       if (xhr.status >= 200 && xhr.status < 300) {
//         const res = JSON.parse(xhr.responseText);
//         setMessage({ type: "success", text: "✅ OMR processed successfully!" });
//         setResults(res);

//         setFiles([]);
//         setAnswerKey("");
//       } else {
//         setMessage({
//           type: "error",
//           text: `❌ Upload failed: ${xhr.status} ${xhr.statusText}`,
//         });
//       }
//     };

//     xhr.onerror = () => {
//       setUploading(false);
//       setMessage({ type: "error", text: "❌ Network error" });
//     };

//     xhr.send(formData);
//   };

//   return (
//     <div className="min-h-screen flex flex-col bg-white p-6">
//       <h3 className="text-2xl font-bold mb-6">Upload & Process OMR Sheets</h3>

//       <div
//         className="border-2 border-dashed p-6 rounded-xl flex flex-col md:flex-row"
//         onDrop={onDrop}
//         onDragOver={(e) => e.preventDefault()}
//       >
//         {/* LEFT SIDE */}
//         <div className="md:w-2/3">
//           <p className="text-gray-700 mb-4">Drag & drop OMR sheets here, or</p>

//           <div className="flex items-center gap-4 mb-4">
//             <input
//               ref={inputRef}
//               type="file"
//               multiple
//               accept="image/*"
//               onChange={onSelectFiles}
//               className="hidden"
//             />

//             <button
//               onClick={() => inputRef.current.click()}
//               className="px-4 py-2 bg-blue-500 text-white rounded-full"
//             >
//               Choose files
//             </button>

//             <button
//               onClick={() => setFiles([])}
//               disabled={!files.length}
//               className="px-4 py-2 bg-gray-200 rounded-full disabled:opacity-50"
//             >
//               Clear
//             </button>

//             <button
//               onClick={uploadFiles}
//               disabled={!files.length || uploading}
//               className="px-4 py-2 bg-green-500 text-white rounded-full disabled:opacity-50"
//             >
//               {uploading ? "Uploading..." : "Send OMR Sheets"}
//             </button>
//           </div>

//           {/* ANSWER KEY */}
//           <label className="block text-gray-700 font-medium">
//             Custom Answer Key (JSON)
//           </label>

//           <textarea
//             value={answerKey}
//             onChange={(e) => setAnswerKey(e.target.value)}
//             placeholder='{"1":"A","2":"C",...}'
//             className="w-full p-2 border rounded-md h-32 text-sm font-mono"
//           />

//           {message && (
//             <div
//               className={`p-3 mt-3 rounded ${
//                 message.type === "success"
//                   ? "bg-green-100 text-green-700"
//                   : "bg-red-100 text-red-700"
//               }`}
//             >
//               {message.text}
//             </div>
//           )}

//           {uploading && (
//             <div className="mt-3">
//               <div className="w-full bg-gray-200 h-3 rounded">
//                 <div
//                   className="h-3 bg-green-500 rounded"
//                   style={{ width: `${progress}%` }}
//                 ></div>
//               </div>
//               <p className="text-sm mt-1">{progress}%</p>
//             </div>
//           )}

//           {results && (
//             <div className="mt-6 p-4 bg-gray-100 rounded-lg max-h-64 overflow-auto">
//               <h4 className="font-semibold mb-2">Results:</h4>
//               <pre className="text-sm">{JSON.stringify(results, null, 2)}</pre>
//             </div>
//           )}
//         </div>

//         {/* RIGHT SIDE FILE LIST */}
//         <div className="md:w-1/3 mt-6 md:mt-0 md:pl-6">
//           <div className="bg-gray-50 p-4 rounded-lg max-h-56 overflow-auto">
//             {files.length === 0 ? (
//               <p className="text-gray-500">No files selected</p>
//             ) : (
//               files.map((f, i) => (
//                 <div
//                   key={i}
//                   className="flex justify-between py-2 border-b last:border-none"
//                 >
//                   <div>
//                     <p className="text-sm">{f.name}</p>
//                     <p className="text-xs text-gray-500">
//                       {(f.size / 1024).toFixed(1)} KB
//                     </p>
//                   </div>

//                   <button
//                     onClick={() => removeFile(i)}
//                     className="text-red-500 text-sm"
//                   >
//                     Remove
//                   </button>
//                 </div>
//               ))
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }