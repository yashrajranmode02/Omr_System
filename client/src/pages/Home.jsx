// // import React, { useState, useRef } from "react";
// // import { Link } from "react-router-dom";
// // import { motion } from "framer-motion";

// // export default function Home() {
// //   const [files, setFiles] = useState([]);
// //   const [uploading, setUploading] = useState(false);
// //   const [progress, setProgress] = useState(0);
// //   const [message, setMessage] = useState(null);
// //   const inputRef = useRef(null);

// //   const handleFiles = (fileList) => {
// //     const arr = Array.from(fileList);
// //     // optional: filter by allowed types and max size
// //     const allowed = arr.filter((f) => f.size <= 10 * 1024 * 1024); // <=10MB each
// //     setFiles((prev) => [...prev, ...allowed]);
// //   };

// //   const onSelectFiles = (e) => {
// //     handleFiles(e.target.files);
// //   };

// //   const onDrop = (e) => {
// //     e.preventDefault();
// //     e.stopPropagation();
// //     if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
// //   };

// //   const onDragOver = (e) => {
// //     e.preventDefault();
// //     e.stopPropagation();
// //   };

// //   const removeFile = (index) => {
// //     setFiles((prev) => prev.filter((_, i) => i !== index));
// //   };

// //   const formatBytes = (bytes) => {
// //     if (bytes === 0) return "0 B";
// //     const k = 1024;
// //     const sizes = ["B", "KB", "MB", "GB", "TB"];
// //     const i = Math.floor(Math.log(bytes) / Math.log(k));
// //     return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
// //   };

// //   const uploadFiles = () => {
// //     if (!files.length) return;
// //     setUploading(true);
// //     setMessage(null);
// //     setProgress(0);

// //     const formData = new FormData();
// //     files.forEach((f, i) => formData.append("files", f));

// //     // Using XMLHttpRequest to track upload progress
// //     const xhr = new XMLHttpRequest();
// //     xhr.open("POST", "/api/omr/upload"); // update endpoint as needed

// //     xhr.upload.onprogress = (e) => {
// //       if (e.lengthComputable) {
// //         const percent = Math.round((e.loaded / e.total) * 100);
// //         setProgress(percent);
// //       }
// //     };

// //     xhr.onload = () => {
// //       setUploading(false);
// //       if (xhr.status >= 200 && xhr.status < 300) {
// //         setMessage({ type: "success", text: "Files uploaded successfully!" });
// //         setFiles([]);
// //       } else {
// //         setMessage({ type: "error", text: `Upload failed: ${xhr.statusText || xhr.status}` });
// //       }
// //     };

// //     xhr.onerror = () => {
// //       setUploading(false);
// //       setMessage({ type: "error", text: "Network error during upload." });
// //     };

// //     xhr.send(formData);
// //   };

// //   return (
// //     <div className="min-h-screen flex flex-col bg-white">
// //       {/* Header */}
// //       <header className="sticky top-0 z-50 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 shadow-lg backdrop-blur-md bg-opacity-90 transition-all duration-300">
// //         <div className="container mx-auto flex justify-between items-center py-4 px-6">
// //           {/* Logo / Title */}
// //           <h1 className="text-3xl pl-10 font-extrabold text-white tracking-wide hover:scale-105 transition-transform duration-300 cursor-pointer">
// //             OMR <span className="text-blue-400">System</span>
// //           </h1>

// //           {/* Buttons */}
// //           <div className="space-x-4 flex items-center">
// //             <Link to="/login">
// //               <button className="px-5 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-md hover:shadow-lg hover:scale-110 transition-all duration-300">
// //                 Login
// //               </button>
// //             </Link>
// //             <Link to="/register">
// //               <button className="px-5 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-md hover:shadow-lg hover:scale-110 transition-all duration-300">
// //                 Sign Up
// //               </button>
// //             </Link>
// //           </div>
// //         </div>
// //       </header>

// //       {/* Hero Section */}
// //       <section className="bg-black pl-29 flex flex-col md:flex-row items-center justify-between container mx-auto px-6  py-16">
// //         {/* Text */}
// //         <motion.div
// //           initial={{ opacity: 0, x: -50 }}
// //           animate={{ opacity: 1, x: 0 }}
// //           transition={{ duration: 0.8 }}
// //           className="md:w-1/2 space-y-6"
// //         >
// //           <h2 className="text-4xl font-extrabold text-white">
// //             Automated <span className="text-blue-500">OMR Evaluation </span>Made Simple
// //           </h2>
// //           <p className="text-lg text-gray-400">
// //             Generate OMR sheets, scan them instantly, and get real-time results. Streamline your
// //             evaluation process with our advanced technology.
// //           </p>
// //           <button className="px-6 py-3 bg-gradient-to-r from-blue-500 to-green-500 text-white rounded-lg shadow-md hover:opacity-90">
// //             Get Started
// //           </button>
// //         </motion.div>

// //         {/* Image */}
// //         <motion.div
// //           initial={{ opacity: 0, x: 50 }}
// //           animate={{ opacity: 1, x: 0 }}
// //           transition={{ duration: 0.8 }}
// //           className="md:w-1/2 mt-8 md:mt-0 flex justify-center"
// //         >
// //           <img
// //             src="../icons/Img1.png"
// //             alt="OMR Illustration"
// //             className="rounded-xl shadow-lg w-96"
// //           />
// //         </motion.div>
// //       </section>

// //       {/* Upload & Send OMR Sheets Section */}
// //       <section className="container mx-auto px-6 py-12">
// //         <h3 className="text-2xl font-bold text-gray-900 mb-6">Upload & Send OMR Sheets</h3>

// //         <div
// //           className="border-2 border-dashed border-gray-300 rounded-xl p-6 flex flex-col md:flex-row items-center justify-between"
// //           onDrop={onDrop}
// //           onDragOver={onDragOver}
// //         >
// //           <div className="md:w-2/3">
// //             <div className="mb-4">
// //               <p className="text-gray-700">Drag & drop scanned OMR sheets here, or</p>
// //             </div>

// //             <div className="flex items-center gap-4">
// //               <input
// //                 ref={inputRef}
// //                 type="file"
// //                 multiple
// //                 accept="image/*,application/pdf"
// //                 onChange={onSelectFiles}
// //                 className="hidden"
// //               />

// //               <button
// //                 onClick={() => inputRef.current && inputRef.current.click()}
// //                 className="px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-sm hover:scale-105 transition-transform"
// //               >
// //                 Choose files
// //               </button>

// //               <button
// //                 onClick={() => setFiles([])}
// //                 disabled={!files.length}
// //                 className="px-4 py-2 bg-gray-100 text-gray-800 rounded-full shadow-sm disabled:opacity-50"
// //               >
// //                 Clear
// //               </button>

// //               <button
// //                 onClick={uploadFiles}
// //                 disabled={!files.length || uploading}
// //                 className="ml-auto px-4 py-2 bg-gradient-to-r from-green-500 to-teal-500 text-white rounded-full shadow-md disabled:opacity-50"
// //               >
// //                 {uploading ? "Uploading..." : "Send OMR Sheets"}
// //               </button>
// //             </div>

// //             {/* Message & Progress */}
// //             <div className="mt-4">
// //               {message && (
// //                 <div className={`p-3 rounded ${message.type === "success" ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"}`}>
// //                   {message.text}
// //                 </div>
// //               )}

// //               {uploading && (
// //                 <div className="w-full bg-gray-200 h-3 rounded mt-3">
// //                   <div className="h-3 rounded" style={{ width: `${progress}%`, background: "linear-gradient(90deg,#10b981,#06b6d4)" }} />
// //                   <p className="text-sm text-gray-600 mt-1">{progress}%</p>
// //                 </div>
// //               )}
// //             </div>
// //           </div>

// //           {/* File preview list */}
// //           <div className="md:w-1/3 mt-6 md:mt-0 md:pl-6">
// //             <div className="bg-gray-50 p-4 rounded-lg shadow-sm max-h-56 overflow-auto">
// //               {files.length === 0 ? (
// //                 <p className="text-gray-500">No files selected</p>
// //               ) : (
// //                 files.map((f, i) => (
// //                   <div key={i} className="flex items-center justify-between py-2 border-b last:border-b-0">
// //                     <div>
// //                       <p className="text-sm font-medium text-gray-800">{f.name}</p>
// //                       <p className="text-xs text-gray-500">{formatBytes(f.size)}</p>
// //                     </div>
// //                     <div className="flex items-center gap-2">
// //                       <button onClick={() => removeFile(i)} className="text-sm text-red-500">Remove</button>
// //                     </div>
// //                   </div>
// //                 ))
// //               )}
// //             </div>
// //           </div>
// //         </div>

// //         <p className="text-sm text-gray-500 mt-3">Tip: You can upload multiple scanned sheets at once (max ~10MB per file). Update <code>/api/omr/upload</code> to match your backend.</p>
// //       </section>

// //       {/* Features Section */}
// //       <section className="bg-black py-16 px-6 pt-0.5">
// //         <div className="container mx-auto text-center pt-9">
// //           <h3 className="text-3xl font-bold text-white mb-10">
// //             Everything You Need for OMR Evaluation
// //           </h3>
// //           <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
// //             {[
// //               { icon: "📄", title: "Generate OMR Sheets", desc: "Download customizable OMR templates for your exams with various question formats and layouts." },
// //               { icon: "📤", title: "Upload Scanned Sheets", desc: "Process multiple scanned answer sheets at once with our batch upload and processing system." },
// //               { icon: "⚡", title: "Instant Results", desc: "Get accurate evaluation results in seconds with detailed analytics and performance insights." },
// //             ].map((item, i) => (
// //               <motion.div
// //                 key={i}
// //                 initial={{ opacity: 0, y: 50 }}
// //                 animate={{ opacity: 1, y: 0 }}
// //                 transition={{ duration: 0.6, delay: i * 0.3 }}
// //                 className="bg-gray-100 p-8 rounded-2xl shadow-md transform transition duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer"
// //               >
// //                 <div className="text-5xl mb-4">{item.icon}</div>
// //                 <h4 className="text-xl font-semibold mb-2">{item.title}</h4>
// //                 <p className="text-gray-600">{item.desc}</p>
// //               </motion.div>
// //             ))}
// //           </div>
// //         </div>
// //       </section>

// //       {/* Footer */}
// //       <footer className="bg-gray-900 py-6 text-center text-gray-400">
// //         <p>© 2025 OMR System. All rights reserved.</p>
// //       </footer>
// //     </div>
// //   );
// // }
// import React, { useState, useRef } from "react";
// import { Link } from "react-router-dom";
// import { motion } from "framer-motion";

// export default function Home() {
//   const [files, setFiles] = useState([]);
//   const [uploading, setUploading] = useState(false);
//   const [progress, setProgress] = useState(0);
//   const [message, setMessage] = useState(null);
//   const [results, setResults] = useState(null);
//   const inputRef = useRef(null);

//   const handleFiles = (fileList) => {
//     const arr = Array.from(fileList);
//     const allowed = arr.filter((f) => f.size <= 10 * 1024 * 1024); // <=10MB each
//     setFiles((prev) => [...prev, ...allowed]);
//   };

//   const onSelectFiles = (e) => handleFiles(e.target.files);
//   const onDrop = (e) => {
//     e.preventDefault();
//     e.stopPropagation();
//     if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
//   };
//   const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
//   const removeFile = (index) => setFiles((prev) => prev.filter((_, i) => i !== index));
//   const formatBytes = (bytes) => {
//     if (bytes === 0) return "0 B";
//     const k = 1024;
//     const sizes = ["B", "KB", "MB", "GB"];
//     const i = Math.floor(Math.log(bytes) / Math.log(k));
//     return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
//   };

//   const uploadFiles = () => {
//     if (!files.length) return;
//     setUploading(true);
//     setMessage(null);
//     setProgress(0);
//     setResults(null);

//     const formData = new FormData();
//     files.forEach((f) => formData.append("files", f));

//     const xhr = new XMLHttpRequest();
//     xhr.open("POST", "http://localhost:5000/api/omr/upload"); // <-- Correct backend URL

//     xhr.upload.onprogress = (e) => {
//       if (e.lengthComputable) {
//         setProgress(Math.round((e.loaded / e.total) * 100));
//       }
//     };

//     xhr.onload = () => {
//       setUploading(false);
//       if (xhr.status >= 200 && xhr.status < 300) {
//         const res = JSON.parse(xhr.responseText);
//         setMessage({ type: "success", text: "Files uploaded successfully!" });
//         setResults(res.results || null);
//         setFiles([]);
//       } else {
//         setMessage({ type: "error", text: `Upload failed: ${xhr.statusText || xhr.status}` });
//       }
//     };

//     xhr.onerror = () => {
//       setUploading(false);
//       setMessage({ type: "error", text: "Network error during upload." });
//     };

//     xhr.send(formData);
//   };

//   return (
//     <div className="min-h-screen flex flex-col bg-white">
//       {/* Header & Hero Section omitted for brevity */}
      
//       {/* Upload Section */}
//       <section className="container mx-auto px-6 py-12">
//         <h3 className="text-2xl font-bold mb-6">Upload & Send OMR Sheets</h3>

//         <div
//           className="border-2 border-dashed p-6 rounded-xl flex flex-col md:flex-row items-center justify-between"
//           onDrop={onDrop} onDragOver={onDragOver}
//         >
//           <div className="md:w-2/3">
//             <div className="mb-4">
//               <p className="text-gray-700">Drag & drop scanned OMR sheets here, or</p>
//             </div>

//             <div className="flex items-center gap-4">
//               <input
//                 ref={inputRef}
//                 type="file"
//                 multiple
//                 accept="image/*"
//                 onChange={onSelectFiles}
//                 className="hidden"
//               />
//               <button
//                 onClick={() => inputRef.current && inputRef.current.click()}
//                 className="px-4 py-2 bg-blue-500 text-white rounded-full"
//               >
//                 Choose files
//               </button>
//               <button
//                 onClick={() => setFiles([])}
//                 disabled={!files.length}
//                 className="px-4 py-2 bg-gray-200 rounded-full disabled:opacity-50"
//               >
//                 Clear
//               </button>
//               <button
//                 onClick={uploadFiles}
//                 disabled={!files.length || uploading}
//                 className="ml-auto px-4 py-2 bg-green-500 text-white rounded-full disabled:opacity-50"
//               >
//                 {uploading ? "Uploading..." : "Send OMR Sheets"}
//               </button>
//             </div>

//             {message && (
//               <div className={`p-3 mt-4 rounded ${message.type === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
//                 {message.text}
//               </div>
//             )}

//             {uploading && (
//               <div className="w-full bg-gray-200 h-3 rounded mt-3">
//                 <div className="h-3 rounded bg-green-400" style={{ width: `${progress}%` }} />
//                 <p className="text-sm mt-1">{progress}%</p>
//               </div>
//             )}

//             {results && (
//               <div className="mt-6 p-4 bg-gray-100 rounded-lg max-h-64 overflow-auto">
//                 <h4 className="font-semibold mb-2">Results:</h4>
//                 <pre className="text-sm">{JSON.stringify(results, null, 2)}</pre>
//               </div>
//             )}
//           </div>

//           {/* File Preview */}
//           <div className="md:w-1/3 mt-6 md:mt-0 md:pl-6">
//             <div className="bg-gray-50 p-4 rounded-lg max-h-56 overflow-auto">
//               {files.length === 0 ? (
//                 <p className="text-gray-500">No files selected</p>
//               ) : files.map((f, i) => (
//                 <div key={i} className="flex justify-between py-2 border-b last:border-b-0">
//                   <div>
//                     <p className="text-sm font-medium">{f.name}</p>
//                     <p className="text-xs text-gray-500">{formatBytes(f.size)}</p>
//                   </div>
//                   <button onClick={() => removeFile(i)} className="text-red-500 text-sm">Remove</button>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>

//         <p className="text-sm text-gray-500 mt-3">
//           Tip: You can upload multiple scanned sheets at once (max ~10MB each). Backend URL is `/api/omr/upload`.
//         </p>
//       </section>
//     </div>
//   );
// }


import React, { useState, useRef } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function Home() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState(null);
  const [results, setResults] = useState(null);
  const [answerKey, setAnswerKey] = useState(""); // for custom answer key
  const inputRef = useRef(null);

  const handleFiles = (fileList) => {
    const arr = Array.from(fileList);
    const allowed = arr.filter((f) => f.size <= 10 * 1024 * 1024); // max 10MB
    setFiles((prev) => [...prev, ...allowed]);
  };

  const onSelectFiles = (e) => handleFiles(e.target.files);
  const onDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
  };
  const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const removeFile = (index) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const isValidJSON = (str) => {
    try { JSON.parse(str); return true; }
    catch { return false; }
  };

  const uploadFiles = () => {
    if (!files.length) return;

    // validate answer key JSON
    if (answerKey.trim() && !isValidJSON(answerKey)) {
      setMessage({ type: "error", text: "Answer key is not valid JSON." });
      return;
    }

    setUploading(true);
    setMessage(null);
    setProgress(0);
    setResults(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    if (answerKey.trim()) formData.append("answer_key", answerKey);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:8000/process-omr"); // adjust backend URL

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
    };

    xhr.onload = () => {
      setUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        const res = JSON.parse(xhr.responseText);
        setMessage({ type: "success", text: "Files uploaded successfully!" });
        setResults(res.results || null);
        setFiles([]);
        setAnswerKey(""); // clear answer key after upload
      } else {
        setMessage({ type: "error", text: `Upload failed: ${xhr.statusText || xhr.status}` });
      }
    };

    xhr.onerror = () => {
      setUploading(false);
      setMessage({ type: "error", text: "Network error during upload." });
    };

    xhr.send(formData);
  };

  return (
    <div className="min-h-screen flex flex-col bg-white">
      {/* Header omitted for brevity */}

      <section className="container mx-auto px-6 py-12">
        <h3 className="text-2xl font-bold mb-6">Upload & Send OMR Sheets</h3>

        <div
          className="border-2 border-dashed p-6 rounded-xl flex flex-col md:flex-row items-center justify-between"
          onDrop={onDrop} onDragOver={onDragOver}
        >
          <div className="md:w-2/3">
            <div className="mb-4">
              <p className="text-gray-700">Drag & drop scanned OMR sheets here, or</p>
            </div>

            <div className="flex items-center gap-4 mb-4">
              <input
                ref={inputRef}
                type="file"
                multiple
                accept="image/*"
                onChange={onSelectFiles}
                className="hidden"
              />
              <button onClick={() => inputRef.current && inputRef.current.click()} className="px-4 py-2 bg-blue-500 text-white rounded-full">Choose files</button>
              <button onClick={() => setFiles([])} disabled={!files.length} className="px-4 py-2 bg-gray-200 rounded-full disabled:opacity-50">Clear</button>
              <button onClick={uploadFiles} disabled={!files.length || uploading} className="ml-auto px-4 py-2 bg-green-500 text-white rounded-full disabled:opacity-50">
                {uploading ? "Uploading..." : "Send OMR Sheets"}
              </button>
            </div>

            {/* Answer key input */}
            <div className="mb-4">
              <label className="block text-gray-700 mb-1 font-medium">Custom Answer Key (JSON)</label>
              <textarea
                value={answerKey}
                onChange={(e) => setAnswerKey(e.target.value)}
                placeholder='{"1":3,"2":1,...}' 
                className="w-full p-2 border rounded-md h-32 text-sm font-mono"
              />
              <p className="text-xs text-gray-500 mt-1">Optional. Leave empty to use default answer key.</p>
            </div>

            {message && (
              <div className={`p-3 mt-2 rounded ${message.type === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                {message.text}
              </div>
            )}

            {uploading && (
              <div className="w-full bg-gray-200 h-3 rounded mt-3">
                <div className="h-3 rounded bg-green-400" style={{ width: `${progress}%` }} />
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

          {/* File preview */}
          <div className="md:w-1/3 mt-6 md:mt-0 md:pl-6">
            <div className="bg-gray-50 p-4 rounded-lg max-h-56 overflow-auto">
              {files.length === 0 ? (
                <p className="text-gray-500">No files selected</p>
              ) : files.map((f, i) => (
                <div key={i} className="flex justify-between py-2 border-b last:border-b-0">
                  <div>
                    <p className="text-sm font-medium">{f.name}</p>
                    <p className="text-xs text-gray-500">{formatBytes(f.size)}</p>
                  </div>
                  <button onClick={() => removeFile(i)} className="text-red-500 text-sm">Remove</button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <p className="text-sm text-gray-500 mt-3">
          Tip: Upload multiple scanned sheets (max ~10MB each). Backend URL: `/process-omr`.
        </p>
      </section>
    </div>
  );
}
