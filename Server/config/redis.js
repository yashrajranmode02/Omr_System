require('dotenv').config();
const { createClient } = require('redis');

const client = createClient({
  username: 'default',
  password: process.env.REDIS_PASSWORD,
  socket: {
    host: process.env.REDIS_HOST,
    port: process.env.REDIS_PORT
  }
});

// Optional: just handle error
client.on('error', (err) => {
  console.error('❌ Redis Client Error:', err);
});

module.exports = client;
