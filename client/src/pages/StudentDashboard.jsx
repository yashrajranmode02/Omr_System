import React, { useState } from "react";
import axios from "axios";
import toast, { Toaster } from "react-hot-toast";
import DashboardLayout from "../styles/DashboardLayout";

const StudentDashboard = () => {
  const [rollNumber, setRollNumber] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!rollNumber) return toast.error("Please enter a roll number");

    setLoading(true);
    try {
      const token = localStorage.getItem("token");
      const res = await axios.get(`http://localhost:5000/api/results/search?rollNumber=${rollNumber}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResults(res.data);
      setSearched(true);
      if (res.data.length === 0) toast("No results found", { icon: "ℹ️" });
    } catch (err) {
      console.error(err);
      toast.error("Error fetching results");
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <Toaster />
      <h1 className="text-2xl font-bold mb-6">🎓 Student Dashboard</h1>

      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <h2 className="text-xl font-semibold mb-4">Check Results</h2>
        <form onSubmit={handleSearch} className="flex gap-4">
          <input
            type="text"
            placeholder="Enter Roll Number"
            value={rollNumber}
            onChange={(e) => setRollNumber(e.target.value)}
            className="flex-1 px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </form>
      </div>

      {searched && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          {results.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b">
                    <th className="py-2">Test Name</th>
                    <th className="py-2">Score</th>
                    <th className="py-2">Date</th>
                    <th className="py-2">Remarks</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result) => (
                    <tr key={result._id} className="border-b last:border-0">
                      <td className="py-3">{result.testId?.title || result.fileName || "Direct Upload"}</td>
                      <td className="py-3 font-bold text-blue-600">{result.score}</td>
                      <td className="py-3">{new Date(result.createdAt).toLocaleDateString()}</td>
                      <td className="py-3">{result.remarks || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500">No results found for this roll number.</p>
          )}
        </div>
      )}
    </DashboardLayout>
  );
};

export default StudentDashboard;
