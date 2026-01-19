import React, { useEffect, useState } from "react";
import api from "../services/api";
import { useNavigate } from "react-router-dom";
import DashboardLayout from "../styles/DashboardLayout";
import UploadOMR from "./UploadOMR";

const TeacherDashboard = () => {
  const [tests, setTests] = useState([]);
  const [showUploadOMR, setShowUploadOMR] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchTests();
  }, []);

  const fetchTests = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await api.get("/tests", {
        headers: { Authorization: `Bearer ${token}` },
      });
      setTests(res.data);
    } catch (err) {
      console.error("❌ Error fetching tests:", err);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Teacher Dashboard 🧑‍🏫</h1>
      </div>

      <div className="flex gap-4">
        {/* 🔵 Upload Button */}
        <button
          onClick={() => setShowUploadOMR(true)}
          className="
            px-6 
            py-2.5 
            bg-blue-600 
            text-white 
            font-semibold 
            rounded-md 
            hover:bg-blue-700 
            transition 
            duration-200
            "
        >
          Upload OMR Sheets
        </button>

        {/* 🎨 Template Designer Button */}
        <button
          onClick={() => navigate("/teacher/template-designer")}
          className="
            px-6 
            py-2.5 
            bg-purple-600 
            text-white 
            font-semibold 
            rounded-md 
            hover:bg-purple-700 
            transition 
            duration-200
            "
        >
          Design New Template
        </button>
      </div>

      {/* ⭐ Full OMR Upload UI appears below button */}
      {showUploadOMR && (
        <div className="mt-8">
          <UploadOMR />
        </div>
      )}
    </DashboardLayout>
  );
};

export default TeacherDashboard;
