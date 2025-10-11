import React from "react";
import DashboardLayout from "../styles/DashboardLayout";

const StudentDashboard = () => {
  return (
    <DashboardLayout>
      <h1 className="text-2xl font-bold mb-6">🎓 Student Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Join Test */}
        <div className="p-6 bg-white shadow rounded">
          <h2 className="text-xl font-semibold mb-2">📚 Join Test</h2>
          <p className="text-gray-600">Enter a test code to participate.</p>
          <button className="mt-3 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
            Join Now
          </button>
        </div>

        {/* View Results */}
        <div className="p-6 bg-white shadow rounded">
          <h2 className="text-xl font-semibold mb-2">📊 View Results</h2>
          <p className="text-gray-600">See your test performance instantly.</p>
          <button className="mt-3 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
            View Results
          </button>
        </div>

        {/* Profile */}
        <div className="p-6 bg-white shadow rounded">
          <h2 className="text-xl font-semibold mb-2">👤 Profile</h2>
          <p className="text-gray-600">Update your details here.</p>
          <button className="mt-3 bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">
            Edit Profile
          </button>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default StudentDashboard;
