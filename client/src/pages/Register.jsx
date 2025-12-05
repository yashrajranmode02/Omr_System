import { useState } from "react";
import axios from "axios";
import toast, { Toaster } from "react-hot-toast";
import { useNavigate, Link } from "react-router-dom";

export default function Register() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    role: "student",
  });

  const navigate = useNavigate();

  // ✅ FIXED handleChange
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // ✅ FIXED handleSubmit with correct payload
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const payload = {
        name: formData.name,
        email_id: formData.email, // backend expects email_id
        password: formData.password,
        role: formData.role,
      };

      const res = await axios.post("http://localhost:5000/api/auth/register", payload);

      toast.success("Registration Successful 🎉");

      // save token
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("user", JSON.stringify(res.data.user));

      // redirect based on role
      if (res.data.user.role === "student") {
        navigate("/student");
      } else {
        navigate("/teacher");
      }
    } catch (err) {
      toast.error(err.response?.data?.msg || "Registration Failed ❌");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <Toaster position="top-right" />
      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-lg shadow-lg w-96 space-y-6"
      >
        <h2 className="text-2xl font-bold text-center">Register</h2>

        <input
          type="text"
          name="name"
          placeholder="Full Name"
          value={formData.name}
          onChange={handleChange}
          className="w-full px-4 py-2 border rounded-md"
          required
        />

        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleChange}
          className="w-full px-4 py-2 border rounded-md"
          required
        />

        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          className="w-full px-4 py-2 border rounded-md"
          required
        />

        <select
          name="role"
          value={formData.role}
          onChange={handleChange}
          className="w-full px-4 py-2 border rounded-md"
        >
          <option value="student">Student</option>
          <option value="teacher">Teacher</option>
        </select>

        <button
          type="submit"
          className="w-full py-2 bg-gradient-to-r from-green-500 to-blue-600 text-white rounded-md hover:opacity-90 transition"
        >
          Register
        </button>

        <Link to="/login" className="text-blue-500 underline block text-center">
          Already have an account?
        </Link>
      </form>
    </div>
  );
}
