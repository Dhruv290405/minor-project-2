// models/Prediction.js
const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
    patientId: {
        type: String,
        default: 'anonymous_user', // Can be linked to an actual User ID later
    },
    patientName: {
        type: String,
        required: true,
        trim: true,
    },
    vitals: {
        Pregnancies: Number,
        Glucose: Number,
        BloodPressure: Number,
        SkinThickness: Number,
        Insulin: Number,
        BMI: Number,
        Pedigree: Number,
        Age: Number
    },
    result: {
        risk_percentage: Number,
        primary_risk_factors: [String],
        protective_factors: [String]
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Prediction', predictionSchema);