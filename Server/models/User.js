const mongoose=require('mongoose')
const {Schema}=mongoose
const structure1=new Schema({
        name:{
            required:true,
            type:String,
        },
        email_id:{
            required:true,
            type:String,
        },
        passwordHash:{
            required:true,
            type:String
        },
        role:{
            type:String,
            enum:['student','teacher'],
            required:true,
        },
        rollNumber:{
            type:String
        }
},{timestamps:true})
module.exports=structure1