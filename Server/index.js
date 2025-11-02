const express = require('express');
require('dotenv').config();
const { main } = require('./config/db'); 

const app = express();
app.use(express.json());
// const cors = require('cors');
// app.use(cors()); // allow all origins (dev only)

const cors = require('cors');
app.use(cors({
  origin: 'http://localhost:5173',  // allow frontend
  methods: ['GET','POST'],
  credentials: true
}));


app.get('/api/ping', (req, res) => res.json({ ok: true }));
app.use('/api/auth', require('./Routes/Auth'));
app.use('/api/tests', require('./Routes/Test'));
app.use('/api/omr', require('./utils/OmrProcessor'));

async function connection() {
  try {
    // Connect to MongoDB 
    await main();
    console.log('✅ MongoDB connected');

    // Start server
    app.listen(process.env.PORT_No, () => {
      console.log(`🚀 Server is listening at ${process.env.PORT_No}`);
    });

  } catch (err) {
    console.error('❌ Error while connecting to databases:', err);
  }
}

connection();
