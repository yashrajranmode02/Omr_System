const jwt = require('jsonwebtoken'); 
require('dotenv').config();

const requireAuth = async (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ msg: 'No token, auth denied' });

  try {
    const decoded = jwt.verify(token, process.env.SECRET_KEY);
    req.user = decoded; // { id, email_id, role }
    next();
  } catch (err) {
    return res.status(401).json({ msg: 'Invalid token' });
  }
};

const requireRole = (role) => (req, res, next) => {
  if (!req.user) return res.status(401).json({ msg: 'Auth required' });
  if (req.user.role !== role) return res.status(403).json({ msg: 'Forbidden' });
  next();
};

module.exports = { requireAuth, requireRole };
