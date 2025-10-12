import requests
import time
import statistics
import concurrent.futures
from datetime import datetime
import json

class RobustInferenceMonitor:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
        self.session = self._create_retry_session()
    
    def _create_retry_session(self):
        """Create a session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=2,  # Maximum number of retries
            backoff_factor=1,  # Delay between retries
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def single_request_metrics(self, payload):
        """Measure metrics for a single request with better error handling"""
        start_time = time.time()
        try:
            response = self.session.post(
                self.endpoint_url, 
                json=payload, 
                timeout=15,  # Increased timeout
                headers={'Content-Type': 'application/json'}
            )
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            success = response.status_code == 200
            
            response_data = None
            if success:
                try:
                    response_data = response.json()
                except:
                    response_data = {"raw_response": response.text[:200]}  # Truncate if too long
            
            return {
                'timestamp': datetime.now().isoformat(),
                'latency_ms': latency_ms,
                'success': success,
                'status_code': response.status_code,
                'response': response_data,
                'error': None
            }
            
        except requests.exceptions.Timeout:
            end_time = time.time()
            return {
                'timestamp': datetime.now().isoformat(),
                'latency_ms': (end_time - start_time) * 1000,
                'success': False,
                'status_code': 'Timeout',
                'response': None,
                'error': 'Request timed out after 15 seconds'
            }
        except requests.exceptions.ConnectionError:
            end_time = time.time()
            return {
                'timestamp': datetime.now().isoformat(),
                'latency_ms': (end_time - start_time) * 1000,
                'success': False,
                'status_code': 'ConnectionError',
                'response': None,
                'error': 'Could not connect to server'
            }
        except Exception as e:
            end_time = time.time()
            return {
                'timestamp': datetime.now().isoformat(),
                'latency_ms': (end_time - start_time) * 1000,
                'success': False,
                'status_code': 'Exception',
                'response': None,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def run_diagnostic_test(self, payload):
        """Run a simple diagnostic test first"""
        print("🔍 Running diagnostic test...")
        
        # Test basic connectivity
        try:
            health_url = self.endpoint_url.replace('/predict', '/health')
            health_response = requests.get(health_url, timeout=5)
            print(f"✅ Health check: {health_response.status_code}")
            if health_response.status_code == 200:
                print(f"   Response: {health_response.json()}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        # Test single prediction
        print(f"🔍 Testing prediction endpoint...")
        result = self.single_request_metrics(payload)
        
        print(f"   Status: {'✅ Success' if result['success'] else '❌ Failed'}")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        if result['error']:
            print(f"   Error: {result['error']}")
        if result['response']:
            print(f"   Response: {result['response']}")
        
        return result
    
    def run_comprehensive_test(self, payload, duration_seconds=30, requests_per_second=2):
        """Run comprehensive test with better error handling"""
        print(f"🚀 Starting comprehensive test for {duration_seconds} seconds...")
        
        start_time = time.time()
        results = []
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Send fewer concurrent requests to avoid overwhelming the server
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(requests_per_second, 3)) as executor:
                futures = [executor.submit(self.single_request_metrics, payload) 
                          for _ in range(requests_per_second)]
                
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=20)
                        batch_results.append(result)
                    except concurrent.futures.TimeoutError:
                        batch_results.append({
                            'timestamp': datetime.now().isoformat(),
                            'latency_ms': 20000,  # 20 seconds timeout
                            'success': False,
                            'status_code': 'FutureTimeout',
                            'response': None,
                            'error': 'Future timeout exceeded'
                        })
            
            results.extend(batch_results)
            request_count += len(batch_results)
            
            # Progress reporting
            elapsed = time.time() - start_time
            successful = len([r for r in batch_results if r['success']])
            print(f"   Progress: {elapsed:.1f}s | Batch: {successful}/{len(batch_results)} successful")
            
            # Wait to maintain rate
            batch_time = time.time() - batch_start
            if batch_time < 1.0:
                time.sleep(1.0 - batch_time)
        
        return self._calculate_metrics(results)
    
    def _calculate_metrics(self, results):
        """Calculate metrics with safe empty data handling"""
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        latencies = [r['latency_ms'] for r in successful_requests]
        
        # Handle empty results safely
        if not results:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'error_rate_percentage': 100.0,
                'throughput_rps': 0,
                'latency_ms': {'avg': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0},
                'test_duration_seconds': 0,
                'all_requests_failed': True,
                'error_breakdown': {}
            }
        
        # Calculate total time from first to last request
        try:
            start_time = datetime.fromisoformat(results[0]['timestamp'])
            end_time = datetime.fromisoformat(results[-1]['timestamp'])
            total_time = (end_time - start_time).total_seconds()
        except:
            total_time = 0
        
        # Error breakdown
        error_breakdown = {}
        for req in failed_requests:
            error_type = req.get('status_code', 'Unknown')
            error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
        
        metrics = {
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'error_rate_percentage': (len(failed_requests) / len(results)) * 100,
            'throughput_rps': len(results) / total_time if total_time > 0 else 0,
            'latency_ms': {
                'avg': statistics.mean(latencies) if latencies else 0,
                'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (statistics.median(latencies) if latencies else 0),
                'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else (statistics.median(latencies) if latencies else 0),
                'min': min(latencies) if latencies else 0,
                'max': max(latencies) if latencies else 0
            },
            'test_duration_seconds': total_time,
            'all_requests_failed': len(successful_requests) == 0,
            'error_breakdown': error_breakdown
        }
        
        return metrics

# Usage with better diagnostics
print("🎯 ML Inference Monitor - Enhanced Version")
monitor = RobustInferenceMonitor("http://mlops-env.eba-ed55ksnr.us-east-1.elasticbeanstalk.com/predict")
payload = {"text": "Software development and cloud computing services"}

# First run diagnostic test
diagnostic_result = monitor.run_diagnostic_test(payload)

if diagnostic_result['success']:
    print("\n✅ Diagnostic passed! Running comprehensive test...")
    # Run comprehensive test with conservative settings
    results = monitor.run_comprehensive_test(payload, duration_seconds=15, requests_per_second=2)
    print("\n📊 Comprehensive Metrics:")
    print(json.dumps(results, indent=2))
else:
    print(f"\n❌ Diagnostic failed. Fix the endpoint issue first.")
    print(f"   Error: {diagnostic_result.get('error', 'Unknown error')}")