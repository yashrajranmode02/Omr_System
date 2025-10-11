const express = require('express');
require('dotenv').config();
const { main } = require('./config/db');
const RedisClient = require('./config/redis');

const app = express();
app.use(express.json());

app.get('/api/ping', (req, res) => res.json({ ok: true }));
app.use('/api/auth', require('./Routes/Auth'));
app.use('/api/tests', require('./Routes/Test'));

async function connection() {
  try {
    // Connect to MongoDB 
    await main();
    console.log('✅ MongoDB connected');

    // Connect to Redis explicitly
    if (!RedisClient.isOpen) {
      await RedisClient.connect();
      console.log('✅ Redis connected');
    }

    // Start server after both DBs are connected
    app.listen(process.env.PORT_No, () => {
      console.log(`🚀 Server is listening at ${process.env.PORT_No}`);
    });

  } catch (err) {
    console.error('❌ Error while connecting to databases:', err);
  }
}

connection();
