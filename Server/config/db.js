const mongoose=require('mongoose')
const structure1=require('../models/User')

require('dotenv').config()
async function main(){
await mongoose.connect(process.env.MONGO_STRING)
}

const User=mongoose.model('User',structure1);
module.exports={main,User};