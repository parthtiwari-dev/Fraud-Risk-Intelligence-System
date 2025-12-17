# 🐳 FRIS — Docker End-to-End Guide

This document explains **exactly what Docker is doing in FRIS**, whether you should stop containers, and how to run everything step by step from PowerShell. No fluff. No magic.

---

## 1️⃣ Should you stop the running container now?

### Short answer
**Yes, you can stop it safely.**

### Why
- Your **local Docker container** is only for **testing and learning**
- Render will build and run **its own container** from your repo
- Your local container has **zero connection** to Render

Stopping local Docker:
- ❌ does NOT affect Render
- ❌ does NOT delete your image
- ❌ does NOT break anything

### When to keep it running
Only keep it running if you are:
- testing endpoints locally
- experimenting with payloads
- debugging logs

Otherwise, stop it.

---

## 2️⃣ Difference between Image and Container (important)

### Docker Image
- Blueprint
- Built once
- Reusable
- Stored locally

FRIS image:
```
fris-api
```

### Docker Container
- Running instance of image
- Temporary
- Can be stopped/deleted anytime

Example container:
```
happy_ellis
```

You can delete containers freely. Images are what matter.

---

## 3️⃣ How to stop the running container

### Option A — CTRL + C (simplest)
In the terminal where Docker is running:
```
CTRL + C
```

### Option B — Docker Desktop
- Open Docker Desktop
- Find running container
- Click **Stop**

---

## 4️⃣ How to run FRIS with Docker (from scratch)

### Step 1 — Build the image

From project root:
```
docker build -t fris-api .
```

This:
- reads Dockerfile
- installs dependencies
- copies code + model artifacts
- creates an image

---

### Step 2 — Run the container

```
docker run -p 8000:8000 fris-api
```

What this means:
- container port 8000 → local port 8000
- FastAPI listens on 0.0.0.0 inside container

---

### Step 3 — Access the API

Open in browser:
```
http://127.0.0.1:8000/docs
```

Available endpoints:
- `/health`
- `/predict`
- `/explain`

---

## 5️⃣ How to test prediction manually

### Example payload
```json
{
  "Time": 100000,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.5516,
  "V12": -0.617801,
  "V13": -0.99139,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

Use Swagger UI or curl.

---

## 6️⃣ What Render will do differently

Render will:
- pull your GitHub repo
- run `docker build`
- start container on its servers
- expose public HTTPS URL

Your laptop:
- can be OFF
- Docker Desktop can be CLOSED

Render runs independently.

---

## 7️⃣ Final mental model (lock this in)

```
Local Docker → testing & learning
Docker Image → deployment artifact
Render → always-on hosted container
```

Containers are disposable.
Images + code are the product.

---

## 8️⃣ Status

✅ ML artifacts frozen
✅ API stable
✅ Docker verified
✅ Ready for Render

This concludes Docker phase for FRIS.

