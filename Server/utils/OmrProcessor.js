// import fs from "fs";
// import path from "path";
// import axios from "axios";
// import Result from "../models/Result.js";
// import OMRUpload from "../models/OmrUpload.js";

// /**
//  * processFile(filePath, testId, uploadId)
//  * - If process.env.OMR_SERVICE_URL set -> POST file to service and expect JSON result.
//  * - Else -> create a mock Result (for dev/testing).
//  */
// export async function processFile(filePath, testId, uploadId) {
//   try {
//     // If you have a real OMR microservice (FastAPI), configure OMR_SERVICE_URL in .env
//     const serviceUrl = process.env.OMR_SERVICE_URL;
//     if (serviceUrl) {
//       const form = new FormData();
//       form.append("file", fs.createReadStream(filePath));
//       form.append("testId", testId);

//       const headers = form.getHeaders ? form.getHeaders() : {};
//       const resp = await axios.post(`${serviceUrl}/process-omr`, form, { headers });

//       // expected resp.data structure: { rollNumber, answers: { "1":"B", ... }, score, remarks }
//       const { rollNumber, answers, score, remarks } = resp.data;

//       const result = new Result({
//         testId,
//         rollNumber,
//         answers,
//         score,
//         remarks
//       });
//       await result.save();

//       await OMRUpload.findByIdAndUpdate(uploadId, { processed: true, resultId: result._id, studentRoll: rollNumber });

//       return result;
//     } else {
//       // Mock processing (for local dev). Create random answers or a placeholder result.
//       const mockAnswers = {}; // e.g. 10 questions
//       for (let i = 1; i <= 10; i++) mockAnswers[i] = ["A","B","C","D","E"][Math.floor(Math.random()*5)];

//       const mockScore = Math.floor(Math.random() * 101); // 0-100

//       const result = new Result({
//         testId,
//         rollNumber: null,
//         answers: mockAnswers,
//         score: mockScore,
//         remarks: "Mock-processed (dev)"
//       });
//       await result.save();

//       await OMRUpload.findByIdAndUpdate(uploadId, { processed: true, resultId: result._id });

//       return result;
//     }
//   } catch (err) {
//     console.error("OMR processing failed:", err.message || err);
//     throw err;
//   }
// }


// const express = require('express');
// const router = express.Router();
// const multer  = require('multer');
// const fs = require('fs');
// const axios = require('axios');
// const FormData = require('form-data');

// // configure multer
// const upload = multer({ dest: 'tmp_uploads/' }); // temp storage

// router.post('/upload-omr', upload.array('files'), async (req, res) => {
//   try {
//     const files = req.files; // array
//     if (!files || files.length === 0) return res.status(400).json({ msg: 'No files uploaded' });

//     const form = new FormData();
//     files.forEach(file => {
//       form.append('files', fs.createReadStream(file.path), { filename: file.originalname });
//     });

//     // send to FastAPI
//     const response = await axios.post('http://localhost:8000/process-omr', form, {
//       headers: {
//         ...form.getHeaders()
//       },
//       maxContentLength: Infinity,
//       maxBodyLength: Infinity
//     });

//     // cleanup temp files
//     files.forEach(file => fs.unlink(file.path, () => {}));

//     // Save results to MongoDB here (example)
//     const results = response.data;
//     // TODO: insert into DB (use your Result model)
//     // Example: await Result.create({ batchId: results.batch_id, results: results.results, userId: req.user.id, createdAt: new Date() })

//     return res.json(results);

//   } catch (err) {
//     console.error('Error forwarding to FastAPI:', err.response?.data || err.message || err);
//     // cleanup temp files if present
//     if (req.files) req.files.forEach(f => fs.unlink(f.path, () => {}));
//     return res.status(500).json({ msg: 'Error processing OMR', error: err.message });
//   }
// });

// module.exports = router;


const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

// -------------------- Multer Setup --------------------
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // max 10MB per file
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith('image/')) {
      return cb(new Error('Only image files are allowed'), false);
    }
    cb(null, true);
  }
});

// -------------------- Upload Route --------------------
router.post('/upload', upload.array('files'), async (req, res) => {
  try {
    const files = req.files;
    if (!files || files.length === 0) {
      return res.status(400).json({ msg: 'No files uploaded' });
    }

    // Prepare FormData to send to FastAPI
    const form = new FormData();
    files.forEach(file => {
      form.append('files', file.buffer, { filename: file.originalname });
    });

    // Send files to FastAPI backend
    const response = await axios.post('http://localhost:8000/process-omr', form, {
      headers: { ...form.getHeaders() },
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    // Get results from FastAPI
    const results = response.data;

    // Optional: save to MongoDB
    // await Result.create({
    //   batchId: results.batch_id,
    //   results: results.results,
    //   userId: req.user.id,
    //   createdAt: new Date()
    // });

    // Send FastAPI results back to frontend
    return res.json(results);

  } catch (err) {
    console.error('Error sending files to FastAPI:', err.response?.data || err.message || err);
    return res.status(500).json({ msg: 'Error processing OMR', error: err.message });
  }
});

module.exports = router;
