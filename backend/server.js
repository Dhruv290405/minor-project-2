// server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const Prediction = require('./models/Prediction');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
// Note: Replace with your actual MongoDB URI in a .env file later. 
// Using local MongoDB for development right now.
mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log('✅ Connected to MongoDB'))
    .catch(err => console.error('❌ MongoDB Connection Error:', err));

// ---------------------------------------------------------
// THE CORE ORCHESTRATOR ROUTE
// ---------------------------------------------------------
app.post('/api/assess', async (req, res) => {
    try {
        // Separate the patient name from the vitals so the ML engine receives
        // exactly the 8 fields it validates.
        const { patientName, ...vitals } = req.body;

        if (!patientName || !patientName.trim()) {
            return res.status(400).json({
                success: false,
                message: "Patient name is required"
            });
        }

        console.log("1. Received request from Frontend. Patient:", patientName, "Vitals:", vitals);

        // Make the internal API call to the Python Microservice
        // Ensure your FastAPI server is running on port 8000!
        console.log("2. Relaying data to Python ML Engine...");
        const pythonResponse = await axios.post('http://127.0.0.1:8000/predict', vitals);
        
        const mlData = pythonResponse.data;
        console.log("3. Received prediction from ML Engine:", mlData.risk_percentage, "%");

        // Save the entire transaction to MongoDB
        const newRecord = new Prediction({
            patientName: patientName.trim(),
            vitals: vitals,
            result: {
                risk_percentage: mlData.risk_percentage,
                primary_risk_factors: mlData.explanation.primary_risk_factors,
                protective_factors: mlData.explanation.protective_factors
            }
        });

        await newRecord.save();
        console.log("4. Record successfully saved to MongoDB.");

        // Return the final packaged response to the client
        res.status(200).json({
            success: true,
            message: "Assessment complete",
            data: mlData
        });

    } catch (error) {
        console.error("Server Error:", error.message);
        
        // Error handling if the Python server is down
        if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({
                success: false,
                message: "Machine Learning service is currently unavailable."
            });
        }

        res.status(500).json({ success: false, message: "Internal Server Error" });
    }
});

// ---------------------------------------------------------
// HISTORY ROUTE — paginated list of past assessments (10 per page)
// ---------------------------------------------------------
app.get('/api/history', async (req, res) => {
    try {
        const page = Math.max(1, parseInt(req.query.page) || 1);
        const limit = 10; // fixed page size

        const [records, total] = await Promise.all([
            Prediction.find().sort({ createdAt: -1 }).skip((page - 1) * limit).limit(limit),
            Prediction.countDocuments()
        ]);

        res.status(200).json({
            success: true,
            data: records,
            pagination: {
                page,
                limit,
                total,
                totalPages: Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error("History Error:", error.message);
        res.status(500).json({ success: false, message: "Internal Server Error" });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Node.js API Gateway running on http://localhost:${PORT}`);
});