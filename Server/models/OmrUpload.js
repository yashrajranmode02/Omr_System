import mongoose from "mongoose";

const omrUploadSchema = new mongoose.Schema({
  testId: { type: mongoose.Schema.Types.ObjectId, ref: "Test", required: true },
  uploadedBy: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  originalName: { type: String },
  fileName: { type: String },
  filePath: { type: String },
  processed: { type: Boolean, default: false },
  resultId: { type: mongoose.Schema.Types.ObjectId, ref: "Result", default: null },
  studentRoll: { type: String, default: null } // optional, filled by OMR processor if it can parse roll
}, { timestamps: true });

export default mongoose.model("OMRUpload", omrUploadSchema);
