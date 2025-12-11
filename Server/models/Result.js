const mongoose = require("mongoose");

const ResultSchema = new mongoose.Schema({
  testId: { type: mongoose.Schema.Types.ObjectId, ref: "Test" }, // Made optional for direct OMR uploads
  rollNumber: { type: String, required: true },
  score: { type: Number, required: true },
  answers: { type: Object },
  remarks: { type: String },
  fileName: { type: String },
  detected: { type: Object },
  error: { type: String },
  processingTime: { type: Number }
}, { timestamps: true });

module.exports = mongoose.model("Result", ResultSchema);
