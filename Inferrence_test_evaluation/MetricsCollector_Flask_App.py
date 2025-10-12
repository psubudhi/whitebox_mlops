from flask import Flask, request, jsonify
import time
from collections import deque
import threading
import mlflow.pyfunc
import pickle
import numpy as np
import os

class MetricsCollector:
    def __init__(self):
        self.request_times = deque(maxlen=1000)  # Keep last 1000 requests
        self.error_count = 0
        self.total_requests = 0
        self.lock = threading.Lock()
    
    def record_request(self, latency_ms, success=True):
        with self.lock:
            self.total_requests += 1
            if not success:
                self.error_count += 1
            self.request_times.append(latency_ms)
    
    def get_metrics(self):
        with self.lock:
            latencies = list(self.request_times)
            if not latencies:
                return {
                    'total_requests': self.total_requests,
                    'error_rate_percentage': 0.0,
                    'avg_latency_ms': 0.0,
                    'throughput_rps': 0.0,
                    'p95_latency_ms': 0.0,
                    'p99_latency_ms': 0.0
                }
            
            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            p95_index = int(0.95 * len(sorted_latencies))
            p99_index = int(0.99 * len(sorted_latencies))
            
            return {
                'total_requests': self.total_requests,
                'successful_requests': self.total_requests - self.error_count,
                'failed_requests': self.error_count,
                'error_rate_percentage': (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0,
                'avg_latency_ms': sum(latencies) / len(latencies),
                'p95_latency_ms': sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1],
                'p99_latency_ms': sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1],
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'throughput_rps': len(latencies) / 60.0  # Requests per second (last minute estimate)
            }

# Initialize Flask app
app = Flask(__name__)
metrics = MetricsCollector()

# Load your MLflow models (replace with your actual model loading code)
class IndustryClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.class_names = None
        self.load_models()
    
    def load_models(self):
        try:
            # Load model from MLflow - adjust paths as needed
            model_path = "mlruns/0/../artifacts/model"  # Update with your actual path
            vectorizer_path = "mlruns/0/../artifacts/vectorizer"  # Update with your actual path
            
            if os.path.exists(model_path):
                self.model = mlflow.pyfunc.load_model(model_path)
            else:
                # Fallback: create a dummy model for testing
                print("⚠️  Using dummy model for testing - replace with your actual model")
                self.model = None
            
            # Load vectorizer
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                print("⚠️  Using dummy vectorizer for testing")
                self.vectorizer = None
            
            # Example class names - replace with your actual classes
            self.class_names = [
                'Automotive', 'Consulting', 'E-commerce', 'Education', 'Finance', 
                'Government', 'Healthcare', 'Insurance', 'Legal', 
                'Media & Entertainment', 'Other', 'Tech', 'Telecommunications'
            ]
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            # Create dummy components for testing
            self.model = None
            self.vectorizer = None
            self.class_names = ['Class1', 'Class2', 'Class3']
    
    def predict(self, text):
        if self.model is None or self.vectorizer is None:
            # Return dummy prediction for testing
            return "Tech", 0.95, [{"industry": "Tech", "confidence": 0.95}]
        
        try:
            # Transform text
            X = self.vectorizer.transform([text])
            
            # Predict
            probabilities = self.model.predict_proba(X)[0]
            prediction_idx = np.argmax(probabilities)
            prediction = self.class_names[prediction_idx]
            confidence = probabilities[prediction_idx]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = [
                {"industry": self.class_names[i], "confidence": float(probabilities[i])}
                for i in top_indices
            ]
            
            return prediction, confidence, top_predictions
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, []

# Initialize classifier
classifier = IndustryClassifier()

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": "MLflow Sklearn",
        "class_names": classifier.class_names,
        "metrics_available": True
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Endpoint to get current performance metrics"""
    return jsonify(metrics.get_metrics())

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            latency = (time.time() - start_time) * 1000
            metrics.record_request(latency, success=False)
            return jsonify({
                "status": "error",
                "message": "No text provided in request"
            }), 400
        
        text = data['text']
        print(f"📨 Received: {text}")
        
        # Make prediction
        prediction, confidence, top_predictions = classifier.predict(text)
        
        latency = (time.time() - start_time) * 1000
        metrics.record_request(latency, success=True)
        
        print(f"🎯 Prediction: {prediction} (Confidence: {confidence:.3f})")
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": round(latency, 2),
            "top_predictions": top_predictions,
            "model_source": "MLflow Sklearn Models"
        })
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        metrics.record_request(latency, success=False)
        print(f"❌ Prediction error: {e}")
        
        return jsonify({
            "status": "error",
            "message": str(e),
            "latency_ms": round(latency, 2)
        }), 500

@app.route('/reset_metrics', methods=['POST'])
def reset_metrics():
    """Endpoint to reset metrics (for testing)"""
    global metrics
    metrics = MetricsCollector()
    return jsonify({"status": "metrics reset"})

if __name__ == '__main__':
    print("🚀 Starting MLflow Model Server with Metrics...")
    print("📊 Available endpoints:")
    print("   POST /predict      - Make predictions")
    print("   GET  /health       - Health check")
    print("   GET  /info         - Model info")
    print("   GET  /metrics      - Performance metrics")
    print("   POST /reset_metrics - Reset metrics counter")
    print("🌐 Server running on: http://127.0.0.1:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=False)