const express = require('express');
const router = express.Router();
const { requireAuth, requireRole } = require('../middleware/auth');
const Test = require('../models/Test');
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const Result = require("../models/Result");

const storage = multer.memoryStorage();
const upload = multer({ storage });

// Create test (teacher only)
router.post('/', requireAuth, requireRole('teacher'), async (req, res) => {
  const { title, templateType, answerKey, assignedStudents } = req.body;
  try {
    const test = new Test({
      teacherId: req.user.id,
      title,
      templateType,
      answerKey,
      assignedStudents
    });
    await test.save();
    res.json(test);
  } catch (err) {
    console.error(err);
    res.status(500).json({ msg: 'Server error' });
  }
});

// Get tests for teacher or assigned to student
router.get('/', requireAuth, async (req, res) => {
  try {
    if (req.user.role === 'teacher') {
      const tests = await Test.find({ teacherId: req.user.id }).populate('assignedStudents','name email rollNumber');
      return res.json(tests);
    } else {
      // student: tests where assignedStudents includes them
      const tests = await Test.find({ assignedStudents: req.user.id }).populate('teacherId','name email');
      return res.json(tests);
    }
  } catch (err) {
    console.error(err);
    res.status(500).json({ msg: 'Server error' });
  }
});




router.post("/:id/upload", requireAuth, requireRole("teacher"), upload.array("files"), async (req, res) => {
  try {
    if (!req.files?.length) return res.status(400).json({ msg: "No files uploaded" });

    const form = new FormData();
    req.files.forEach(f => form.append("files", f.buffer, { filename: f.originalname }));

    const fastRes = await axios.post("http://localhost:8000/process-omr", form, {
      headers: form.getHeaders(),
    });

    const processed = fastRes.data;

    const savedResults = [];
    for (const r of processed) {
      const doc = await Result.create({
        testId: req.params.id,
        rollNumber: r.rollNumber,
        score: r.score,
        answers: r.answers,
        remarks: r.remarks || "",
      });
      savedResults.push(doc);
    }

    res.json({ msg: "Processed & Saved", savedResults });
  } catch (err) {
    console.error(err);
    res.status(500).json({ msg: "Upload failed" });
  }
});



module.exports = router;
