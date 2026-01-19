import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from "react-router-dom";
import { Suspense, lazy, useEffect } from "react";
import api from "./services/api"; // Add this import

const AuthRedirect = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const checkUsers = async () => {
      try {
        const res = await api.get("/auth/check-users");
        if (res.data.count === 0) {
          if (location.pathname !== "/register") {
            navigate("/register");
          }
        } else {
          // If users exist, and we are at root, redirect to login if not authenticated
          const token = localStorage.getItem("token");
          if (!token && location.pathname === "/") {
            navigate("/login");
          }
        }
      } catch (err) {
        console.error("Error checking users:", err);
      }
    };
    checkUsers();
  }, [navigate, location]);

  return null;
};

const UploadOMR = lazy(() => import("./pages/UploadOMR"));
const Login = lazy(() => import("./pages/Login"));
const Register = lazy(() => import("./pages/Register"));
const TeacherDashboard = lazy(() => import("./pages/TeacherDashboard"));
const StudentDashboard = lazy(() => import("./pages/StudentDashboard"));
const TestDetails = lazy(() => import("./pages/TestDetails"));
const TemplateDesigner = lazy(() => import("./pages/TemplateDesigner"));

import ProtectedRoute from "./components/ProtectedRoute";

export default function App() {
  return (
    <BrowserRouter>
      <AuthRedirect />
      <Suspense fallback={<div className="p-4">Loading...</div>}>
        <Routes>
          {/* Public */}
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Student-only */}
          <Route
            path="/student"
            element={
              <ProtectedRoute role="student">
                <StudentDashboard />
              </ProtectedRoute>
            }
          />

          {/* Teacher-only */}
          <Route
            path="/teacher"
            element={
              <ProtectedRoute role="teacher">
                <TeacherDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/template-designer"
            element={
              <ProtectedRoute role="teacher">
                <TemplateDesigner />
              </ProtectedRoute>
            }
          />

          {/* Both logged-in roles allowed */}
          <Route
            path="/result"
            element={
              <ProtectedRoute>
                <TestDetails />
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
