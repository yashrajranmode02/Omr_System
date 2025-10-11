const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const {User} = require('../config/db');
const { requireAuth } = require('../middleware/auth');
const RedisClient = require('../config/redis');
// Register (teacher or student)
router.post('/register', async (req, res) => {
  const { name, email_id, password, role, rollNumber } = req.body;
  if (!name || !email_id || !password || !role) return res.status(400).json({ msg: 'Missing fields' });
  try {
    let user = await User.findOne({ email_id });
    if (user) return res.status(400).json({ msg: 'User exists' });
    const salt = await bcrypt.genSalt(10);
    const passwordHash = await bcrypt.hash(password, salt);
    user = new User({ name, email_id,passwordHash, role, rollNumber });
    await user.save();
    const token = jwt.sign({ id: user._id, email_id: user.email_id, role: user.role }, process.env.SECRETE_KEY, { expiresIn: '12h' });
    res.json({ token, user: { id: user._id, name: user.name, email_id: user.email_id, role: user.role } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ msg: 'Server error' });
  }
});

// Login
router.post('/login', async (req, res) => {
  const { email_id, password } = req.body;
  if (!email_id || !password) return res.status(400).json({ msg: 'Missing fields' });
  try {
    const user = await User.findOne({ email_id });
    if (!user) return res.status(400).json({ msg: 'Invalid credentials' });
    const isMatch = await bcrypt.compare(password, user.passwordHash);
    if (!isMatch) return res.status(400).json({ msg: 'Invalid credentials' });
    const token = jwt.sign({ id: user._id, email_id: user.email_id, role: user.role }, process.env.SECRETE_KEY, { expiresIn: '12h' });
    res.json({ token, user: { id: user._id, name: user.name, email_id: user.email_id, role: user.role } });
  } catch (err) {
  console.error('Login error:', err);  // Log actual error
  res.status(500).json({ msg: err.message }); // Send error message
}
});
// Logout
router.post('/logout', requireAuth, async (req, res) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    if (!token) return res.status(400).json({ msg: 'No token provided' });

    // Blacklist token for 12h (same as JWT expiry)
    await RedisClient.setEx(`blacklist_${token}`, 12 * 60 * 60, 'true');

    return res.json({ msg: 'Logged out successfully' });
  } catch (err) {
    console.error('Logout Error:', err);
    return res.status(500).json({ msg: 'Server error during logout' });
  }
});

module.exports = router;
