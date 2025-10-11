import { Navigate } from "react-router-dom";

export default function ProtectedRoute({ children, role }) {
  const token = localStorage.getItem("token");
  const user = JSON.parse(localStorage.getItem("user")); // we stored user data in login

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  // If role is required but user role doesn't match → block
  if (role && user?.role !== role) {
    return <Navigate to="/" replace/>;
  }

  return children;
}
