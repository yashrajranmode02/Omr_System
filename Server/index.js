const express = require('express');
require('dotenv').config();
const { main } = require('./config/db');

const app = express();

// MUST COME AFTER app is created
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const cors = require('cors');
app.use(cors({
  origin: 'http://localhost:5173',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  credentials: true
}));


app.get('/api/ping', (req, res) => res.json({ ok: true }));
app.use('/api/auth', require('./Routes/Auth'));
app.use('/api/tests', require('./Routes/Test'));
// app.use('/api/omr', require('./utils/OmrProcessor'));
app.use('/api/results', require('./Routes/Result'));

async function connection() {
  try {
    await main();
    console.log('✅ MongoDB connected');

    app.listen(process.env.PORT_No, () => {
      console.log(`🚀 Server is listening at ${process.env.PORT_No}`);
    });

  } catch (err) {
    console.error('❌ Error while connecting to databases:', err);
  }
}

connection();
