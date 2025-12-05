const express = require("express");
const router = express.Router();
const Result = require("../models/Result");
const { requireAuth } = require("../middleware/auth");

router.get("/search", requireAuth, async (req, res) => {
  try {
    const { rollNumber } = req.query;
    if (!rollNumber) return res.status(400).json({ msg: "rollNumber required" });

    const data = await Result.find({ rollNumber })
      .populate("testId", "title createdAt templateType");

    res.json(data);
  } catch (err) {
    res.status(500).json({ msg: "Search failed" });
  }
});

module.exports = router;
