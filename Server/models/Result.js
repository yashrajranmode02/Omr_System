const mongoose = require('mongoose');

const resultSchema = new mongoose.Schema({
            testId: { 
                        type: mongoose.Schema.Types.ObjectId, ref: 'Test', required: true
                    },
            studentId: { 
                        type: mongoose.Schema.Types.ObjectId, ref: 'User'
                       },
            rollNumber: { type: String },
            answers: { 
                        type: Map, of: String }, // "1"->"B", etc.
            score: { 
                    type: Number },
            remarks: { type: String },
}, { timestamps: true });
resultSchema= mongoose.model('Result', resultSchema);
module.exports =resultSchema;
