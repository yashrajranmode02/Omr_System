
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

    
    return res.json(results);

  } catch (err) {
    console.error('Error sending files to FastAPI:', err.response?.data || err.message || err);
    return res.status(500).json({ msg: 'Error processing OMR', error: err.message });
  }
});

module.exports = router;
