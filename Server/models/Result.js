const mongoose = require("mongoose");

const ResultSchema = new mongoose.Schema({
  testId: { type: mongoose.Schema.Types.ObjectId, ref: "Test", required: true },
  rollNumber: { type: String, required: true },
  score: { type: Number, required: true },
  answers: { type: Object },
  remarks: { type: String },
}, { timestamps: true });

module.exports = mongoose.model("Result", ResultSchema);
