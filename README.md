Firstly create 1 .env file in Server folder:-
inside that paste this.
PORT_No=5000
SECRETE_KEY=yashraj1234
MONGO_STRING="mongodb+srv://Learner_Mongo:Godsplan@cluster0.peanuas.mongodb.net/EDAI"

For venv make sure you have installed:-
pip install fastapi uvicorn
pip install opencv-python
pip install ultralytics
pip install python-multipart


To run the application we required 3 terminal 
in First terminal 
PS C:\Users\yashr\OneDrive\Documents\GitHub\Omr_System/omr_backend
.\venv\Scripts\activate
uvicorn main:app --reload

Make sure you have node.js installed.
Second temrinal
Npm install express
Npm install mongoose
PS C:\Users\yashr\OneDrive\Documents\GitHub\Omr_System/Server
node index.js
Third terminal
PS C:\Users\yashr\OneDrive\Documents\GitHub\Omr_System/client
npm install
npm run dev
