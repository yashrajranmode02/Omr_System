import React, { useState, useEffect } from "react";
import axios from "axios";
import DashboardLayout from "../styles/DashboardLayout";

// Use the Python backend URL
// Use the Python backend URL
const getOmrBaseUrl = () => {
    let url = import.meta.env.VITE_OMR_API_URL || "http://localhost:8000";
    if (url.endsWith("/")) return url.slice(0, -1);
    return url;
};
const FASTAPI_BASE_URL = getOmrBaseUrl();

const TemplateDesigner = () => {
    const [configJson, setConfigJson] = useState("{}");
    const [previewImage, setPreviewImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);
    const [photo, setPhoto] = useState(null);

    // Load default config on mount
    useEffect(() => {
        fetchDefaultConfig();
    }, []);

    const fetchDefaultConfig = async () => {
        try {
            const res = await axios.get(`${FASTAPI_BASE_URL}/default-config`);
            // Format as nice JSON string
            setConfigJson(JSON.stringify(res.data, null, 2));
        } catch (err) {
            console.error("Failed to load default config", err);
            setError("Could not load default template configuration.");
        }
    };

    const handleJsonChange = (e) => {
        setConfigJson(e.target.value);
        setError(null);
        setSuccessMsg(null);
    };

    const handlePhotoChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setPhoto(e.target.files[0]);
        }
    };

    const generatePreview = async () => {
        setLoading(true);
        setError(null);
        setSuccessMsg(null);
        setPreviewImage(null);

        try {
            // Validate JSON locally
            let parsedConfig;
            try {
                parsedConfig = JSON.parse(configJson);
            } catch (e) {
                throw new Error("Invalid JSON syntax.");
            }

            // We send { config: ... } as JSON body
            const res = await axios.post(
                `${FASTAPI_BASE_URL}/preview-config`,
                { config: parsedConfig },
                { responseType: "blob" }
            );

            const imageUrl = URL.createObjectURL(res.data);
            setPreviewImage(imageUrl);
            setSuccessMsg("Preview generated successfully!");
        } catch (err) {
            console.error(err);
            setError(err.message || "Failed to generate preview.");
        } finally {
            setLoading(false);
        }
    };

    const downloadFinalTemplate = async () => {
        setLoading(true);
        setError(null);
        setSuccessMsg(null);

        try {
            // Validate JSON
            try {
                JSON.parse(configJson);
            } catch (e) {
                throw new Error("Invalid JSON syntax.");
            }

            const formData = new FormData();
            formData.append("config_json", configJson);
            if (photo) {
                formData.append("photo", photo);
            }

            const res = await axios.post(`${FASTAPI_BASE_URL}/generate`, formData, {
                responseType: "blob",
            });

            // Create download link
            const url = window.URL.createObjectURL(new Blob([res.data]));
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute("download", "omr_template.png");
            document.body.appendChild(link);
            link.click();
            link.remove();

            setSuccessMsg("Template downloaded!");
        } catch (err) {
            console.error(err);
            setError("Failed to download final template.");
        } finally {
            setLoading(false);
        }
    };


    const downloadJson = () => {
        try {
            const blob = new Blob([configJson], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute("download", "template.json");
            document.body.appendChild(link);
            link.click();
            link.remove();
            setSuccessMsg("JSON downloaded successfully!");
        } catch (e) {
            setError("Failed to download JSON.");
        }
    };

    const copyToClipboard = () => {
        navigator.clipboard.writeText(configJson);
        setSuccessMsg("JSON copied to clipboard!");
        setTimeout(() => setSuccessMsg(null), 2000);
    };

    return (
        <div className="min-h-screen bg-gray-50 p-6 flex flex-col">
            <div className="mb-6 flex justify-between items-center">
                <h1 className="text-3xl font-bold text-gray-800">OMR Template Designer 🎨</h1>
                <button
                    onClick={() => window.history.back()}
                    className="text-gray-600 hover:text-gray-900 font-medium"
                >
                    &larr; Back
                </button>
            </div>

            <div className="flex flex-col lg:flex-row gap-6 h-full">
                {/* LEFT COLUMN: EDITOR */}
                <div className="lg:w-1/2 flex flex-col gap-4">
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 flex-1 flex flex-col">
                        <div className="flex justify-between items-center mb-2">
                            <label className="font-semibold text-gray-700">JSON Configuration</label>
                            <div className="flex gap-2">
                                <button
                                    onClick={fetchDefaultConfig}
                                    className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-600"
                                >
                                    Reset to Default
                                </button>
                                <button
                                    onClick={downloadJson}
                                    className="text-xs px-2 py-1 bg-purple-50 hover:bg-purple-100 rounded text-purple-600"
                                >
                                    Download JSON
                                </button>
                                <button
                                    onClick={copyToClipboard}
                                    className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 rounded text-blue-600"
                                >
                                    Copy JSON
                                </button>
                            </div>
                        </div>

                        <textarea
                            className="w-full flex-1 p-3 font-mono text-sm border rounded-md focus:ring-2 focus:ring-blue-500 outline-none resize-none bg-gray-50"
                            value={configJson}
                            onChange={handleJsonChange}
                            spellCheck="false"
                            style={{ minHeight: "500px" }}
                        />
                    </div>




                    <div className="flex gap-4">
                        <button
                            onClick={generatePreview}
                            disabled={loading}
                            className="flex-1 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-lg shadow transition disabled:opacity-50"
                        >
                            {loading ? "Processing..." : "🔄 Generate Preview"}
                        </button>

                        <button
                            onClick={downloadFinalTemplate}
                            disabled={loading}
                            className="flex-1 py-3 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg shadow transition disabled:opacity-50"
                        >
                            ⬇ Download High-Res PNG
                        </button>
                    </div>

                    {error && (
                        <div className="p-3 bg-red-100 text-red-700 rounded-md text-sm border border-red-200">
                            ❌ {error}
                        </div>
                    )}

                    {successMsg && (
                        <div className="p-3 bg-green-100 text-green-700 rounded-md text-sm border border-green-200">
                            ✅ {successMsg}
                        </div>
                    )}
                </div>

                {/* RIGHT COLUMN: PREVIEW */}
                <div className="lg:w-1/2 bg-gray-200 rounded-xl border border-gray-300 flex items-center justify-center p-4 relative overflow-hidden min-h-[600px]">
                    {previewImage ? (
                        <div className="relative w-full h-full flex justify-center items-center">
                            <img
                                src={previewImage}
                                alt="OMR Preview"
                                className="max-w-full max-h-full shadow-2xl border border-gray-400 object-contain"
                            />
                        </div>
                    ) : (
                        <div className="text-center text-gray-500">
                            <p className="text-xl font-medium">No Preview Available</p>
                            <p className="text-sm">Click "Generate Preview" to see your design.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default TemplateDesigner;
