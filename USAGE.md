
# Shakespearean LLM API

A lightweight Character-Level RNN (Recurrent Neural Network) trained on Shakespeare's corpus, capable of generating stylistic text via a REST API.

This project is fully containerized using **Docker** and exposed via **FastAPI**, ready for production-like environments.

---

## Quick Start

### Prerequisites
* Docker installed on your machine.
* Git.

### 1. Clone the Repository
First, clone:
```bash
git clone git@github.com:ju4nv1e1r4/shakespearean-character-level-rnn.git
```
Then:
```bash
cd shakespearean-character-level-rnn
```

### 2. Build the Docker Image

Build the container image containing the model artifacts and the API server:

```bash
docker build -t shakespearean-lm-api .
```

### 3. Run the Container

Start the API on port 8000:

```bash
docker run -p 8000:8000 shakespearean-lm-api
```

*(The API will be available at `http://localhost:8000`)*

---

## Usage

### Option A: Swagger UI

The easiest way to interact with the model is through the automatic documentation provided by FastAPI.

1. Open your browser and navigate to:
**http://localhost:8000/docs**
2. Click on the **`POST /generate`** endpoint.
3. Click **"Try it out"**.
4. Edit the Request body (JSON) and click **"Execute"**.

### Option B: Command Line (cURL)

You can generate text directly from your terminal:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "The King hath",
  "temperature": 0.6,
  "length": 300
}'
```

---

## Parameters Explained

When sending a request, you can tweak the following parameters to change the output style:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `string` | "The king" | The starting text for the model to complete. |
| `length` | `int` | 200 | Number of characters to generate. |
| `temperature` | `float` | 0.6 | Controls randomness/creativity: <br>

<br>• **< 0.5**: Conservative, repetitive, grammatically stricter. <br>

<br>• **0.5 - 0.8**: Balanced (Recommended). <br>

<br>• **> 1.0**: Chaotic, creative, prone to "hallucinations".

---

## Health Check

To verify if the API is running correctly:

```bash
curl http://localhost:8000/health

```