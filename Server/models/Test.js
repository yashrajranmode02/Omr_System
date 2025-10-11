const mongoose = require('mongoose');
const {Schema}=require('mongoose')
const testSchema = new Schema({
  teacherId: {
             type: mongoose.Schema.Types.ObjectId,
             ref: 'User',
             required: true 
            },
  title:     { 
               type: String,
               required: true 
            },
  templateType: 
            { 
                type: String, 
                enum: ['standard1','standard2','standard3','custom'], 
                default: 'standard1' 
            },
   answerKey:
            { type: Map, of: String 

            }, // question index -> 'A'|'B'...
  assignedStudents: 
            [{ type: mongoose.Schema.Types.ObjectId, 
                ref: 'User' 
            }],
  meta:     { type: Object } // optional metadata (date, duration etc.)

}, { timestamps: true });

const Test= mongoose.model('Test', testSchema);
module.exports=Test;
