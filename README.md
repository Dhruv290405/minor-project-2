# Minor Project 2 - Health Risk Assessment

A full-stack health risk assessment application. Frontend sends patient vitals → Node.js API gateway relays them to a Python ML engine → prediction is saved to MongoDB.

## Architecture

```
┌──────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Frontend │───▶ │ Node.js Gateway   │───▶ │Python ML Engine │
│ (Vite+   │
│ React)│  │      │ (Express :5000)  │      │ (FastAPI :8000) │
└──────────┘      └────────┬─────────┘      └─────────────────┘
                           │
                           ▼
                     ┌──────────┐
                     │ MongoDB  │
                     │ (:27018) │
                     └──────────┘
```

## Project Structure

```
minor-project-2/
├── backend/                  # Express API gateway
│   ├── server.js             # Routes: POST /api/assess, GET /api/history
│   ├── models/Prediction.js  # Mongoose schema for assessment records
│   ├── docker-compose.yml    # MongoDB container (port 27018)
│   ├── .env                  # MONGO_URI connection string
│   └── package.json
├── Frontend/                 # React + TypeScript + Vite + Tailwind
│   ├── src/                  # React source code
│   ├── index.html
│   ├── vite.config.ts        # Dev proxy: /api → localhost:5000
│   ├── tsconfig*.json
│   ├── tailwind.config.js
│   ├── eslint.config.js
│   └── package.json
├── .gitignore
└── README.md
```

## Prerequisites

- **Node.js** (v18+)
- **Docker** (for MongoDB)
- **Python ML Engine** running on `http://127.0.0.1:8000/predict`

## Getting Started

### 1. Start MongoDB

```bash
cd backend
docker compose up -d
```

### 2. Start the Backend

```bash
cd backend
npm install
node server.js
```

### 3. Start the Frontend

```bash
cd Frontend
npm install
npm run dev
```

### 4. Python ML Engine

Ensure your FastAPI ML service is running on `http://127.0.0.1:8000` with a `POST /predict` endpoint.

## API Endpoints

| Method | Endpoint          | Description                          |
| ------ | ----------------- | ------------------------------------ |
| POST   | `/api/assess`     | Submit patient vitals for assessment |
| GET    | `/api/history`    | Paginated list of past assessments (10/page, use `?page=`) |

### POST /api/assess

**Request body:**
```json
{
  "patientName": "John Doe",
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 80,
  "SkinThickness": 25,
  "Insulin": 85,
  "BMI": 28.5,
  "Pedigree": 0.5,
  "Age": 35
}
```

**Response:**
```json
{
  "success": true,
  "message": "Assessment complete",
  "data": {
    "risk_percentage": 45.2,
    "explanation": {
      "primary_risk_factors": ["High glucose", "Elevated BMI"],
      "protective_factors": ["Normal blood pressure"]
    }
  }
}
```
