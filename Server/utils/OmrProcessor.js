// const express = require('express');
// const router = express.Router();
// const multer = require('multer');
// const axios = require('axios');
// const FormData = require('form-data');

// const storage = multer.memoryStorage();

// const upload = multer({
//   storage,
//   limits: { fileSize: 10 * 1024 * 1024 },
//   fileFilter: (req, file, cb) => {
//     if (!file.mimetype.startsWith('image/')) {
//       return cb(new Error('Only image files allowed'), false);
//     }
//     cb(null, true);
//   }
// });

// router.post("/upload", upload.any(), async (req, res) => {
//     const files = req.files.filter(f => f.fieldname === "files");
//     const answerKey = req.body.answer_key; // NOW WORKS

//     if (!files.length) return res.status(400).json({ msg: "No files uploaded" });

//     const form = new FormData();

//     files.forEach(f => {
//         form.append("files", f.buffer, { filename: f.originalname });
//     });

//     form.append("answer_key", answerKey || "{}");

//     const response = await axios.post("http://localhost:8000/process-omr",
//         form,
//         { headers: form.getHeaders() }
//     );

//     return res.json(response.data);
// });



// // module.exports = router;
// const express = require("express");
// const router = express.Router();
// const multer = require("multer");
// const axios = require("axios");
// const FormData = require("form-data");

// // -------------------- Multer Setup --------------------
// const storage = multer.memoryStorage();
// const upload = multer({ storage }).any(); // accept ANY field exactly like frontend sends

// // -------------------- Upload Route --------------------
// router.post("/upload", upload, async (req, res) => {
//   try {
//     console.log("📥 Incoming OMR Body:", req.body);
//     console.log("📸 Incoming OMR Files:", req.files?.length);

//     // Extract answer key from body
//     const answerKey = req.body?.answer_key || "{}";

//     // Extract only image files
//     const files = req.files?.filter(f => f.fieldname === "files");

//     if (!files || files.length === 0) {
//       return res.status(400).json({ msg: "No files uploaded" });
//     }

//     // Prepare FormData for FastAPI
//     const form = new FormData();

//     // 🟢 Add answer key first — VERY IMPORTANT
//     form.append("answer_key", answerKey);

//     console.log("➡️ Forwarding Answer Key to FastAPI:", answerKey);

//     // 🟢 Add images
//     files.forEach(file => {
//       form.append("files", file.buffer, file.originalname);
//     });

//     // Send to FastAPI
//     const response = await axios.post(
//       "http://localhost:8000/process-omr",
//       form,
//       {
//         headers: form.getHeaders(),
//         maxContentLength: Infinity,
//         maxBodyLength: Infinity,
//       }
//     );

//     console.log("✅ FastAPI Response Received");
//     return res.json(response.data);

//   } catch (err) {
//     console.error(
//       "❌ Error in forwarding OMR to FastAPI:",
//       err.response?.data || err.message
//     );

//     return res.status(500).json({
//       msg: "Error processing OMR",
//       error: err.message,
//     });
//   }
// });

// module.exports = router;
const express = require("express");
const router = express.Router();
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

// -------------------- Multer Setup --------------------
const storage = multer.memoryStorage();

// ✅ CORRECT: create multer instance first, THEN call .any()
const upload = multer({ storage }).any();

// -------------------- Upload Route --------------------
router.post("/upload", upload, async (req, res) => {
  try {
    console.log("📥 Incoming Body:", req.body);
    console.log("📸 Incoming Files Count:", req.files?.length);

    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ msg: "No files received by Node server" });
    }

    const files = req.files;
    const answerKey = req.body.answer_key || "{}";

    const form = new FormData();

    // ✅ 반드시 먼저 answer_key 추가
    form.append("answer_key", answerKey);

    // ✅ 모든 이미지 FastAPI로 전달
    files.forEach(file => {
      form.append("files", file.buffer, file.originalname);
    });

    console.log("➡️ Forwarding", files.length, "files to FastAPI");

    const omrUrl = process.env.OMR_API_URL || "http://localhost:8000";
    const response = await axios.post(
      `${omrUrl}/process-omr`,
      form,
      {
        headers: {
          ...form.getHeaders(),
          "Content-Length": form.getLengthSync(), // ✅ prevents hanging
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 300000, // ✅ 5 min timeout
      }
    );

    console.log("✅ FastAPI Response OK");
    return res.json(response.data);

  } catch (err) {
    console.error("❌ OMR Forwarding Error:", err.response?.data || err.message);

    return res.status(500).json({
      msg: "Error forwarding OMR to FastAPI",
      error: err.message,
    });
  }
});

module.exports = router;
