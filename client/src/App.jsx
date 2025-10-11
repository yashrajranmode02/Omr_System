import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import TeacherDashboard from "./pages/TeacherDashboard";
import StudentDashboard from "./pages/StudentDashboard";
import TestDetails from "./pages/TestDetails";
import ProtectedRoute from "./components/ProtectedRoute";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected Routes */}
        <Route
          path="/student"
          element={

            // <ProtectedRoute role="student">
              <StudentDashboard />
            // {/* </ProtectedRoute> */}
          }
        />
        <Route
          path="/teacher"
          element={
            // <ProtectedRoute role="teacher">
              <TeacherDashboard />
            // {/* </ProtectedRoute> */}
          }
        />
        <Route
          path="/result"
          element={
            // <ProtectedRoute>
              <TestDetails />
            // {/* </ProtectedRoute> */}
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
