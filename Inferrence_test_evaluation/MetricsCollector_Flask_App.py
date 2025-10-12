from flask import Flask, request, jsonify
import time
from collections import deque
import threading
import pickle
import numpy as np
import os
import sys

# Add your existing modules to path
sys.path.append('Inferrence_test_evaluation')

class MetricsCollector:
    def __init__(self):
        self.request_times = deque(maxlen=1000)
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
                'throughput_rps': len(latencies) / 60.0
            }

# Initialize Flask app
app = Flask(__name__)
metrics = MetricsCollector()

class IndustryClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.class_names = None
        self.load_models()
    
    def load_models(self):
        try:
            # Try multiple possible model locations based on your structure
            possible_model_paths = [
                "models/model.pkl",
                "Inferrence_test_evaluation/models/model.pkl",
                "model.pkl"
            ]
            
            possible_vectorizer_paths = [
                "models/vectorizer.pkl",
                "Inferrence_test_evaluation/models/vectorizer.pkl", 
                "vectorizer.pkl"
            ]
            
            # Find and load model
            model_loaded = False
            for model_path in possible_model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"✅ Model loaded from {model_path}")
                    model_loaded = True
                    break
            
            # Find and load vectorizer
            vectorizer_loaded = False
            for vectorizer_path in possible_vectorizer_paths:
                if os.path.exists(vectorizer_path):
                    with open(vectorizer_path, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    print(f"✅ Vectorizer loaded from {vectorizer_path}")
                    vectorizer_loaded = True
                    break
            
            if not model_loaded or not vectorizer_loaded:
                print("⚠️  Some model files not found. Using dummy mode.")
                if not model_loaded:
                    self.model = None
                if not vectorizer_loaded:
                    self.vectorizer = None
            
            # Use the class names from your industry classification
            self.class_names = [
                'Automotive', 'Consulting', 'E-commerce', 'Education', 'Finance', 
                'Government', 'Healthcare', 'Insurance', 'Legal', 
                'Media & Entertainment', 'Other', 'Tech', 'Telecommunications'
            ]
            
            print("✅ Industry classifier initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
            self.vectorizer = None
            self.class_names = ['Tech', 'Finance', 'Healthcare']  # Fallback
    
    def predict(self, text):
        # Dummy implementation if models not loaded
        if self.model is None or self.vectorizer is None:
            # Simple rule-based fallback
            text_lower = text.lower()
            if any(word in text_lower for word in ['tech', 'software', 'computer', 'cloud']):
                return "Tech", 0.92, [{"industry": "Tech", "confidence": 0.92}]
            elif any(word in text_lower for word in ['bank', 'finance', 'investment', 'money']):
                return "Finance", 0.88, [{"industry": "Finance", "confidence": 0.88}]
            elif any(word in text_lower for word in ['hospital', 'health', 'medical', 'care']):
                return "Healthcare", 0.85, [{"industry": "Healthcare", "confidence": 0.85}]
            else:
                return "Other", 0.75, [{"industry": "Other", "confidence": 0.75}]
        
        try:
            # Real prediction with loaded models
            X = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(X)[0]
            prediction_idx = np.argmax(probabilities)
            prediction = self.class_names[prediction_idx]
            confidence = probabilities[prediction_idx]
            
            # Get top 3 predictions
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
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "model_loaded": classifier.model is not None,
        "vectorizer_loaded": classifier.vectorizer is not None
    })

@app.route('/info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": "Industry Classification",
        "class_names": classifier.class_names,
        "model_loaded": classifier.model is not None,
        "vectorizer_loaded": classifier.vectorizer is not None,
        "total_classes": len(classifier.class_names)
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
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
        
        # Make prediction
        prediction, confidence, top_predictions = classifier.predict(text)
        
        latency = (time.time() - start_time) * 1000
        metrics.record_request(latency, success=True)
        
        response = {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": round(latency, 2),
            "top_predictions": top_predictions,
            "model_used": "real_model" if classifier.model is not None else "rule_based_fallback"
        }
        
        return jsonify(response)
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        metrics.record_request(latency, success=False)
        return jsonify({
            "status": "error",
            "message": str(e),
            "latency_ms": round(latency, 2)
        }), 500

@app.route('/reset_metrics', methods=['POST'])
def reset_metrics():
    global metrics
    metrics = MetricsCollector()
    return jsonify({"status": "metrics reset"})

@app.route('/test_predictions', methods=['GET'])
def test_predictions():
    """Test endpoint with sample predictions"""
    test_cases = [
        "Software development and cloud computing services",
        "Banking and investment services for retail customers", 
        "Hospital and healthcare services with emergency care",
        "Car manufacturing and automotive parts supplier"
    ]
    
    results = []
    for text in test_cases:
        prediction, confidence, top_predictions = classifier.predict(text)
        results.append({
            "input": text,
            "prediction": prediction,
            "confidence": confidence,
            "top_choices": top_predictions
        })
    
    return jsonify({"test_results": results})

if __name__ == '__main__':
    print("🚀 Starting Industry Classification Server")
    print("📊 Available endpoints:")
    print("   POST /predict        - Make industry predictions")
    print("   GET  /health         - Health check") 
    print("   GET  /info           - Model information")
    print("   GET  /metrics        - Performance metrics")
    print("   GET  /test_predictions - Test with sample inputs")
    print("   POST /reset_metrics  - Reset metrics counter")
    print("🏭 Industry classes:", classifier.class_names)
    print("🌐 Server running on: http://127.0.0.1:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=False)