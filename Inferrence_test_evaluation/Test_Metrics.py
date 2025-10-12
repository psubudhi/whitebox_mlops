import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor

def test_various_inputs():
    """Test with various industry descriptions"""
    test_cases = [
        "Software development and cloud computing services for businesses",
        "Banking and investment services for retail customers", 
        "Hospital and healthcare services with emergency care",
        "Car manufacturing and automotive parts supplier",
        "Online retail store with fast delivery",
        "Law firm specializing in corporate law",
        "Movie production company and streaming service"
    ]
    
    url = "http://127.0.0.1:8000/predict"
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {text}")
        
        payload = {"text": text}
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {result['prediction']} ({result['confidence_percentage']}) - {latency:.1f}ms")
                
                # Show top 3 predictions
                for pred in result['top_predictions'][:2]:
                    print(f"      ↳ {pred['industry']}: {pred['confidence_percentage']}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

def stress_test(duration=30, rps=10):
    """Run stress test for specified duration"""
    url = "http://127.0.0.1:8000/predict"
    payload = {"text": "Software development and cloud computing services"}
    
    print(f"\n🔥 Stress Test: {duration}s at {rps} RPS")
    
    def worker():
        requests_made = 0
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                requests.post(url, json=payload, timeout=5)
                requests_made += 1
                time.sleep(1.0 / rps)  # Maintain rate
            except:
                pass
        return requests_made
    
    # Start multiple workers
    with ThreadPoolExecutor(max_workers=rps) as executor:
        results = list(executor.map(lambda _: worker(), range(rps)))
    
    total_requests = sum(results)
    actual_rps = total_requests / duration
    
    print(f"📈 Stress Test Results:")
    print(f"   Target RPS: {rps}")
    print(f"   Actual RPS: {actual_rps:.1f}")
    print(f"   Total requests: {total_requests}")
    print(f"   Duration: {duration}s")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Metrics Test Suite")
    
    # Test various inputs
    test_various_inputs()
    
    # Check current metrics
    print("\n📊 Current Performance Metrics:")
    metrics = requests.get("http://127.0.0.1:8000/metrics").json()
    print(f"   Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"   P95 Latency: {metrics['p95_latency_ms']:.1f}ms") 
    print(f"   Error Rate: {metrics['error_rate_percentage']:.1f}%")
    print(f"   Throughput: {metrics['throughput_rps']:.1f} RPS")
    print(f"   Total Requests: {metrics['total_requests']}")
    
    # Run stress test
    stress_test(duration=10, rps=5)
    
    # Final metrics
    print("\n📋 Final Metrics:")
    final_metrics = requests.get("http://127.0.0.1:8000/metrics").json()
    print(f"   Total Requests: {final_metrics['total_requests']}")
    print(f"   Error Rate: {final_metrics['error_rate_percentage']:.1f}%")
    print(f"   Avg Latency: {final_metrics['avg_latency_ms']:.1f}ms")