from flask import Flask, jsonify, request, Response
import psycopg2
import redis
import time
import random
import os
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    REGISTRY,
    CONTENT_TYPE_LATEST,
)

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["endpoint"]
)


# Database connection
def get_db():
    try:
        return psycopg2.connect(os.getenv("DATABASE_URL"))
    except Exception as e:
        print(f"DB connection error: {e}")
        return None


# Redis connection
def get_redis():
    try:
        return redis.from_url(os.getenv("REDIS_URL"))
    except Exception as e:
        print(f"Redis connection error: {e}")
        return None


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint - MUST return correct Content-Type"""
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/", status=200).inc()
    return jsonify({"status": "healthy", "service": "metrics-webapp"})


@app.route("/light")
@REQUEST_LATENCY.labels(endpoint="/light").time()
def light():
    """Light endpoint - minimal CPU usage"""
    REQUEST_COUNT.labels(method="GET", endpoint="/light", status=200).inc()
    result = sum(range(100))
    return jsonify({"endpoint": "light", "result": result})


@app.route("/medium")
@REQUEST_LATENCY.labels(endpoint="/medium").time()
def medium():
    """Medium endpoint - moderate CPU usage"""
    REQUEST_COUNT.labels(method="GET", endpoint="/medium", status=200).inc()
    result = sum(i**2 for i in range(1000))
    time.sleep(random.uniform(0.05, 0.15))
    return jsonify({"endpoint": "medium", "result": result})


@app.route("/heavy")
@REQUEST_LATENCY.labels(endpoint="/heavy").time()
def heavy():
    """Heavy endpoint - intensive computation + I/O"""
    REQUEST_COUNT.labels(method="GET", endpoint="/heavy", status=200).inc()

    # CPU intensive
    result = sum(i**2 for i in range(10000))

    # Database query
    conn = get_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
        except Exception as e:
            print(f"DB error: {e}")

    # Redis operation
    r = get_redis()
    if r:
        try:
            r.set(f"key_{time.time()}", result, ex=60)
        except Exception as e:
            print(f"Redis error: {e}")

    time.sleep(random.uniform(0.1, 0.5))
    return jsonify({"endpoint": "heavy", "result": result})


@app.route("/error")
def error():
    """Endpoint that returns errors"""
    REQUEST_COUNT.labels(method="GET", endpoint="/error", status=500).inc()
    return jsonify({"error": "Simulated error"}), 500


if __name__ == "__main__":
    # Initialize some metrics on startup
    print("Starting webapp with Prometheus metrics on /metrics")
    app.run(host="0.0.0.0", port=5000, debug=False)
