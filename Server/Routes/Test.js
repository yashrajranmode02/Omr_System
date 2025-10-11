const express = require('express');
const router = express.Router();
const { requireAuth, requireRole } = require('../middleware/auth');
const Test = require('../models/Test');

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

module.exports = router;
