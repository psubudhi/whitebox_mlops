import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from datetime import datetime
import sys

class HeavyLoadTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': [],
            'start_time': None,
            'end_time': None,
            'error_breakdown': {}
        }
        self.lock = threading.Lock()
        self.stop_test = False
    
    def worker(self, worker_id, duration, request_rate):
        """Worker thread that sends requests at specified rate"""
        latencies = []
        successful = 0
        failed = 0
        error_breakdown = {}
        
        test_texts = [
            "Software development and cloud computing services for businesses",
            "Banking and investment services for retail customers",
            "Hospital and healthcare services with emergency care",
            "Car manufacturing and automotive parts supplier",
            "Online retail store with fast delivery",
            "Law firm specializing in corporate law",
            "Movie production company and streaming service",
            "Insurance company providing life and health coverage",
            "University offering undergraduate and graduate programs",
            "Telecommunications provider with 5G network"
        ]
        
        end_time = time.time() + duration
        request_interval = 1.0 / request_rate
        
        while time.time() < end_time and not self.stop_test:
            start_request = time.time()
            
            # Rotate through test texts
            text = test_texts[self.results['total_requests'] % len(test_texts)]
            payload = {"text": text}
            
            try:
                response = requests.post(
                    f"{self.base_url}/predict", 
                    json=payload, 
                    timeout=10
                )
                latency = (time.time() - start_request) * 1000
                
                if response.status_code == 200:
                    successful += 1
                    latencies.append(latency)
                else:
                    failed += 1
                    error_type = f"HTTP_{response.status_code}"
                    error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
                    
            except requests.exceptions.Timeout:
                failed += 1
                error_breakdown["Timeout"] = error_breakdown.get("Timeout", 0) + 1
            except requests.exceptions.ConnectionError:
                failed += 1
                error_breakdown["ConnectionError"] = error_breakdown.get("ConnectionError", 0) + 1
            except Exception as e:
                failed += 1
                error_type = type(e).__name__
                error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
            
            # Maintain request rate
            elapsed = time.time() - start_request
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)
        
        # Update global results
        with self.lock:
            self.results['total_requests'] += successful + failed
            self.results['successful_requests'] += successful
            self.results['failed_requests'] += failed
            self.results['latencies'].extend(latencies)
            
            for error_type, count in error_breakdown.items():
                self.results['error_breakdown'][error_type] = self.results['error_breakdown'].get(error_type, 0) + count
        
        return successful, failed, latencies
    
    def monitor_metrics(self, duration):
        """Monitor system metrics during test"""
        print(f"\n📊 Starting real-time monitoring for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.stop_test:
            try:
                response = requests.get(f"{self.base_url}/metrics", timeout=5)
                if response.status_code == 200:
                    metrics = response.json()
                    print(f"\r⏱️  Live: {metrics['total_requests']} req | "
                          f"Latency: {metrics['avg_latency_ms']:.1f}ms | "
                          f"Errors: {metrics['error_rate_percentage']:.1f}% | "
                          f"RPS: {metrics['throughput_rps']:.1f}",
                          end="", flush=True)
                time.sleep(2)
            except:
                print("\r❌ Cannot connect to metrics endpoint", end="", flush=True)
                time.sleep(2)
        
        print()  # New line after monitoring
    
    def run_load_test(self, duration=300, concurrent_workers=20, requests_per_second_per_worker=5):
        """Run the heavy load test"""
        print("🚀 STARTING HEAVY LOAD TEST")
        print("=" * 60)
        print(f"⏱️  Duration: {duration} seconds (5 minutes)")
        print(f"👥 Concurrent Workers: {concurrent_workers}")
        print(f"📨 Target Rate: {concurrent_workers * requests_per_second_per_worker} RPS")
        print(f"🎯 Total Target Requests: {duration * concurrent_workers * requests_per_second_per_worker:,}")
        print("=" * 60)
        
        # Reset metrics before test
        try:
            requests.post(f"{self.base_url}/reset_metrics", timeout=5)
            print("✅ Metrics reset successfully")
        except:
            print("⚠️  Could not reset metrics (endpoint might not exist)")
        
        self.results['start_time'] = datetime.now()
        start_time = time.time()
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=self.monitor_metrics, args=(duration,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [
                executor.submit(self.worker, i, duration, requests_per_second_per_worker)
                for i in range(concurrent_workers)
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Worker error: {e}")
        
        self.results['end_time'] = datetime.now()
        test_duration = time.time() - start_time
        
        # Calculate final metrics
        self.calculate_final_metrics(test_duration)
        
        return self.results
    
    def calculate_final_metrics(self, test_duration):
        """Calculate comprehensive metrics from test results"""
        latencies = self.results['latencies']
        
        if latencies:
            sorted_latencies = sorted(latencies)
            self.results['latency_metrics'] = {
                'avg_ms': statistics.mean(latencies),
                'p50_ms': statistics.median(latencies),
                'p75_ms': sorted_latencies[int(0.75 * len(sorted_latencies))] if len(sorted_latencies) >= 4 else statistics.median(latencies),
                'p90_ms': sorted_latencies[int(0.90 * len(sorted_latencies))] if len(sorted_latencies) >= 10 else statistics.median(latencies),
                'p95_ms': sorted_latencies[int(0.95 * len(sorted_latencies))] if len(sorted_latencies) >= 20 else statistics.median(latencies),
                'p99_ms': sorted_latencies[int(0.99 * len(sorted_latencies))] if len(sorted_latencies) >= 100 else statistics.median(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'std_dev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
        else:
            self.results['latency_metrics'] = {}
        
        self.results['throughput_metrics'] = {
            'actual_rps': self.results['total_requests'] / test_duration,
            'success_rps': self.results['successful_requests'] / test_duration,
            'test_duration_seconds': test_duration
        }
        
        self.results['reliability_metrics'] = {
            'error_rate_percentage': (self.results['failed_requests'] / self.results['total_requests']) * 100 if self.results['total_requests'] > 0 else 0,
            'success_rate_percentage': (self.results['successful_requests'] / self.results['total_requests']) * 100 if self.results['total_requests'] > 0 else 0,
            'availability_percentage': 100 - ((self.results['failed_requests'] / self.results['total_requests']) * 100) if self.results['total_requests'] > 0 else 100
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 HEAVY LOAD TEST REPORT - 5 MINUTES")
        print("=" * 70)
        
        print(f"\n⏰ TEST DURATION:")
        print(f"   Start Time: {self.results['start_time']}")
        print(f"   End Time: {self.results['end_time']}")
        print(f"   Duration: {self.results['throughput_metrics']['test_duration_seconds']:.2f} seconds")
        
        print(f"\n📈 THROUGHPUT PERFORMANCE:")
        print(f"   Total Requests: {self.results['total_requests']:,}")
        print(f"   Successful: {self.results['successful_requests']:,}")
        print(f"   Failed: {self.results['failed_requests']:,}")
        print(f"   Actual RPS: {self.results['throughput_metrics']['actual_rps']:.2f}")
        print(f"   Success RPS: {self.results['throughput_metrics']['success_rps']:.2f}")
        
        print(f"\n🎯 RELIABILITY METRICS:")
        print(f"   Success Rate: {self.results['reliability_metrics']['success_rate_percentage']:.3f}%")
        print(f"   Error Rate: {self.results['reliability_metrics']['error_rate_percentage']:.3f}%")
        print(f"   Availability: {self.results['reliability_metrics']['availability_percentage']:.3f}%")
        
        if self.results['error_breakdown']:
            print(f"   Error Breakdown:")
            for error_type, count in self.results['error_breakdown'].items():
                percentage = (count / self.results['total_requests']) * 100
                print(f"     - {error_type}: {count} ({percentage:.2f}%)")
        
        if self.results['latency_metrics']:
            print(f"\n⚡ LATENCY PERFORMANCE (ms):")
            print(f"   Average: {self.results['latency_metrics']['avg_ms']:.2f}ms")
            print(f"   P50 (Median): {self.results['latency_metrics']['p50_ms']:.2f}ms")
            print(f"   P75: {self.results['latency_metrics']['p75_ms']:.2f}ms")
            print(f"   P90: {self.results['latency_metrics']['p90_ms']:.2f}ms")
            print(f"   P95: {self.results['latency_metrics']['p95_ms']:.2f}ms")
            print(f"   P99: {self.results['latency_metrics']['p99_ms']:.2f}ms")
            print(f"   Min: {self.results['latency_metrics']['min_ms']:.2f}ms")
            print(f"   Max: {self.results['latency_metrics']['max_ms']:.2f}ms")
            print(f"   Std Dev: {self.results['latency_metrics']['std_dev_ms']:.2f}ms")
        
        print(f"\n📊 PERFORMANCE ASSESSMENT:")
        self.performance_assessment()
    
    def performance_assessment(self):
        """Provide performance assessment based on metrics"""
        error_rate = self.results['reliability_metrics']['error_rate_percentage']
        avg_latency = self.results['latency_metrics'].get('avg_ms', 0)
        
        # Assess reliability
        if error_rate == 0:
            reliability = "⭐ EXCELLENT - 100% Reliability"
        elif error_rate < 0.1:
            reliability = "⭐ VERY GOOD - High Reliability"
        elif error_rate < 1:
            reliability = "✅ GOOD - Production Ready"
        elif error_rate < 5:
            reliability = "⚠️ ACCEPTABLE - Needs Monitoring"
        else:
            reliability = "❌ POOR - Requires Investigation"
        
        # Assess latency
        if avg_latency < 10:
            latency_rating = "⭐ EXCELLENT - Sub-10ms Performance"
        elif avg_latency < 50:
            latency_rating = "⭐ VERY GOOD - Real-time Ready"
        elif avg_latency < 100:
            latency_rating = "✅ GOOD - Production Suitable"
        elif avg_latency < 500:
            latency_rating = "⚠️ ACCEPTABLE - Monitor Closely"
        else:
            latency_rating = "❌ POOR - Performance Issues"
        
        # Assess throughput
        actual_rps = self.results['throughput_metrics']['actual_rps']
        if actual_rps > 200:
            throughput_rating = "⭐ OUTSTANDING - High Scalability"
        elif actual_rps > 100:
            throughput_rating = "⭐ EXCELLENT - Good Scalability"
        elif actual_rps > 50:
            throughput_rating = "✅ GOOD - Adequate Capacity"
        elif actual_rps > 10:
            throughput_rating = "⚠️ LIMITED - Consider Scaling"
        else:
            throughput_rating = "❌ POOR - Capacity Issues"
        
        print(f"   Reliability: {reliability}")
        print(f"   Latency: {latency_rating}")
        print(f"   Throughput: {throughput_rating}")
        
        # Overall assessment
        if error_rate == 0 and avg_latency < 50 and actual_rps > 50:
            print(f"\n🎯 OVERALL: ✅ PRODUCTION READY - Excellent Performance")
        elif error_rate < 1 and avg_latency < 100 and actual_rps > 20:
            print(f"\n🎯 OVERALL: ✅ PRODUCTION SUITABLE - Good Performance")
        elif error_rate < 5 and avg_latency < 500:
            print(f"\n🎯 OVERALL: ⚠️ ACCEPTABLE - Needs Optimization")
        else:
            print(f"\n🎯 OVERALL: ❌ REQUIRES IMPROVEMENT - Investigate Issues")

def main():
    """Main function to run the heavy load test"""
    try:
        # Test connection first
        print("🔍 Testing connection to inference service...")
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Service is reachable and healthy")
        else:
            print("❌ Service health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("💡 Make sure your Flask app is running on http://127.0.0.1:8000")
        return
    
    # Configuration
    TEST_DURATION = 300  # 5 minutes in seconds
    CONCURRENT_WORKERS = 25  # Increased concurrency
    REQUESTS_PER_SECOND_PER_WORKER = 4  # Reduced per worker to avoid overloading
    
    print(f"\n🎯 Test Configuration:")
    print(f"   Duration: {TEST_DURATION} seconds (5 minutes)")
    print(f"   Concurrent Workers: {CONCURRENT_WORKERS}")
    print(f"   Requests per Worker: {REQUESTS_PER_SECOND_PER_WORKER} RPS")
    print(f"   Target Total: {CONCURRENT_WORKERS * REQUESTS_PER_SECOND_PER_WORKER} RPS")
    print(f"   Expected Requests: {TEST_DURATION * CONCURRENT_WORKERS * REQUESTS_PER_SECOND_PER_WORKER:,}")
    
    input("\n⚠️  Press Enter to start the 5-minute heavy load test (or Ctrl+C to cancel)...")
    
    # Run the test
    tester = HeavyLoadTester()
    
    try:
        results = tester.run_load_test(
            duration=TEST_DURATION,
            concurrent_workers=CONCURRENT_WORKERS,
            requests_per_second_per_worker=REQUESTS_PER_SECOND_PER_WORKER
        )
        
        # Generate report
        tester.generate_report()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"heavy_load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            results_serializable = results.copy()
            results_serializable['start_time'] = results['start_time'].isoformat()
            results_serializable['end_time'] = results['end_time'].isoformat()
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        tester.stop_test = True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()