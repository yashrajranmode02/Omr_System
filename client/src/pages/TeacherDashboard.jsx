import React, { useEffect, useState } from "react";
import axios from "axios";
import DashboardLayout from "../styles/DashboardLayout";

const TeacherDashboard = () => {
  const [tests, setTests] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [showUpload, setShowUpload] = useState(null); // testId for upload modal
  const [newTest, setNewTest] = useState({ name: "", subject: "", duration: "" });
  const [selectedFiles, setSelectedFiles] = useState([]);

  useEffect(() => {
    fetchTests();
  }, []);

  const fetchTests = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await axios.get("http://localhost:5000/api/tests", {
        headers: { Authorization: `Bearer ${token}` },
      });
      setTests(res.data);
    } catch (err) {
      console.error("❌ Error fetching tests:", err);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFiles(e.target.files);
  };

  const handleUploadSheets = async (testId) => {
    if (!selectedFiles.length) {
      alert("Please select files first");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("sheets", selectedFiles[i]);
    }

    try {
      const token = localStorage.getItem("token");
      await axios.post(
        `http://localhost:5000/api/tests/${testId}/upload`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      alert("Sheets uploaded successfully!");
      setShowUpload(null);
      setSelectedFiles([]);
    } catch (err) {
      console.error("❌ Upload error:", err);
    }
  };

  return (
    <DashboardLayout>
      <h1 className="text-3xl font-bold mb-4">Teacher Dashboard 🧑‍🏫</h1>

      {/* Tests Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-6">
        {tests.map((test) => (
          <div
            key={test._id}
            className="border p-4 rounded-xl shadow bg-white hover:shadow-lg transition"
          >
            <h2 className="text-xl font-semibold">{test.name}</h2>
            <p className="text-gray-600">Subject: {test.subject}</p>
            <p className="text-gray-600">Duration: {test.duration} mins</p>
            <p className="text-gray-800 font-bold mt-2">Code: {test.testCode}</p>

            <div className="mt-3 flex space-x-2">
              <button
                onClick={() => setShowUpload(test._id)}
                className="bg-indigo-600 text-white px-3 py-1 rounded hover:bg-indigo-700"
              >
                Upload Sheets
              </button>
              <button className="bg-gray-600 text-white px-3 py-1 rounded hover:bg-gray-700">
                View Results
              </button>
            </div>

            {/* Upload Modal */}
            {showUpload === test._id && (
              <div className="mt-4 p-4 border rounded bg-gray-100">
                <input
                  type="file"
                  multiple
                  onChange={handleFileChange}
                  className="mb-2"
                />
                <button
                  onClick={() => handleUploadSheets(test._id)}
                  className="bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700"
                >
                  Upload Now
                </button>
                <button
                  onClick={() => setShowUpload(null)}
                  className="ml-2 bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </DashboardLayout>
  );
};

export default TeacherDashboard;
