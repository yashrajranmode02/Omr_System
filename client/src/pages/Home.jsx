import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-white">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 shadow-lg backdrop-blur-md bg-opacity-90 transition-all duration-300">
        <div className="container mx-auto flex justify-between items-center py-4 px-6">
          {/* Logo / Title */}
          <h1 className="text-3xl pl-10 font-extrabold text-white tracking-wide hover:scale-105 transition-transform duration-300 cursor-pointer">
            OMR <span className="text-blue-400">System</span>
          </h1>

          {/* Buttons */}
          <div className="space-x-4 flex items-center">
            <Link to="/login">
              <button className="px-5 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-md hover:shadow-lg hover:scale-110 transition-all duration-300">
                Login
              </button>
            </Link>
            <Link to="/register">
              <button className="px-5 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-md hover:shadow-lg hover:scale-110 transition-all duration-300">
                Sign Up
              </button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-black pl-29 flex flex-col md:flex-row items-center justify-between container mx-auto px-6  py-16">
        {/* Text */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="md:w-1/2 space-y-6"
        >
          <h2 className="text-4xl font-extrabold text-white">
            Automated <span className="text-blue-500">OMR Evaluation </span>Made Simple
          </h2>
          <p className="text-lg text-gray-400">
            Generate OMR sheets, scan them instantly, and get real-time results. Streamline your
            evaluation process with our advanced technology.
          </p>
          <button className="px-6 py-3 bg-gradient-to-r from-blue-500 to-green-500 text-white rounded-lg shadow-md hover:opacity-90">
            Get Started
          </button>
        </motion.div>

        {/* Image */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="md:w-1/2 mt-8 md:mt-0 flex justify-center"
        >
          <img
            src="../icons/Img1.png"
            alt="OMR Illustration"
            className="rounded-xl shadow-lg w-96"
          />
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="bg-black py-16 px-6 pt-0.5">
        <div className="container mx-auto text-center pt-9">
          <h3 className="text-3xl font-bold text-white mb-10">
            Everything You Need for OMR Evaluation
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { icon: "📄", title: "Generate OMR Sheets", desc: "Download customizable OMR templates for your exams with various question formats and layouts." },
              { icon: "📤", title: "Upload Scanned Sheets", desc: "Process multiple scanned answer sheets at once with our batch upload and processing system." },
              { icon: "⚡", title: "Instant Results", desc: "Get accurate evaluation results in seconds with detailed analytics and performance insights." },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: i * 0.3 }}
                className="bg-gray-100 p-8 rounded-2xl shadow-md transform transition duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer"
              >
                <div className="text-5xl mb-4">{item.icon}</div>
                <h4 className="text-xl font-semibold mb-2">{item.title}</h4>
                <p className="text-gray-600">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 py-6 text-center text-gray-400">
        <p>© 2025 OMR System. All rights reserved.</p>
      </footer>
    </div>
  );
}
