#!/usr/bin/env python3
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor
import argparse

ENDPOINTS = {
    'light': 'http://localhost:5000/light',
    'medium': 'http://localhost:5000/medium',
    'heavy': 'http://localhost:5000/heavy',
    'error': 'http://localhost:5000/error'
}

def make_request(endpoint):
    """Make a single request"""
    try:
        response = requests.get(ENDPOINTS[endpoint], timeout=5)
        return response.status_code
    except Exception as e:
        return 0

def baseline_load(duration_seconds=3600, rps=10):
    """Steady baseline traffic"""
    print(f"Running baseline load: {rps} req/s for {duration_seconds}s")
    
    end_time = time.time() + duration_seconds
    with ThreadPoolExecutor(max_workers=20) as executor:
        while time.time() < end_time:
            # Distribution: 50% light, 30% medium, 15% heavy, 5% error
            endpoint = random.choices(
                ['light', 'medium', 'heavy', 'error'],
                weights=[50, 30, 15, 5]
            )[0]
            
            executor.submit(make_request, endpoint)
            time.sleep(1.0 / rps)

def spike_pattern(duration_seconds=120, peak_rps=100):
    """Sudden traffic spike"""
    print(f"Running spike pattern: peak {peak_rps} req/s")
    
    # Ramp up
    for rps in range(10, peak_rps, 10):
        baseline_load(duration_seconds=10, rps=rps)
    
    # Peak
    baseline_load(duration_seconds=duration_seconds, rps=peak_rps)
    
    # Ramp down
    for rps in range(peak_rps, 10, -10):
        baseline_load(duration_seconds=10, rps=rps)

def periodic_pattern(duration_seconds=3600):
    """Business hours pattern - high during day, low at night"""
    print("Running periodic pattern (simulated 24h in compressed time)")
    
    # Simulate 24 hours in duration_seconds
    hour_duration = duration_seconds / 24
    
    for hour in range(24):
        if 9 <= hour <= 17:  # Business hours
            rps = 50
        elif 6 <= hour <= 9 or 17 <= hour <= 22:  # Peak edges
            rps = 25
        else:  # Night
            rps = 5
        
        print(f"Hour {hour}: {rps} req/s")
        baseline_load(duration_seconds=int(hour_duration), rps=rps)

def gradual_increase(duration_seconds=7200):
    """Gradual capacity increase"""
    print("Running gradual increase pattern")
    
    steps = 10
    step_duration = duration_seconds / steps
    
    for step in range(steps):
        rps = 10 + (step * 10)
        print(f"Step {step}: {rps} req/s")
        baseline_load(duration_seconds=int(step_duration), rps=rps)

def chaos_pattern(duration_seconds=3600):
    """Random chaos"""
    print("Running chaos pattern")
    
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        rps = random.randint(5, 100)
        duration = random.randint(30, 180)
        print(f"Chaos burst: {rps} req/s for {duration}s")
        baseline_load(duration_seconds=duration, rps=rps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load generator for metrics webapp')
    parser.add_argument('pattern', choices=['baseline', 'spike', 'periodic', 'gradual', 'chaos', 'full'],
                        help='Load pattern to generate')
    parser.add_argument('--duration', type=int, default=3600, help='Duration in seconds')
    
    args = parser.parse_args()
    
    if args.pattern == 'full':
        # Run full week-long scenario (compressed)
        print("=== Starting full load scenario ===")
        baseline_load(duration_seconds=3600)  # 1hr baseline
        spike_pattern(duration_seconds=300)   # 5min spike
        baseline_load(duration_seconds=1800)  # 30min recovery
        periodic_pattern(duration_seconds=3600)  # 1hr periodic
        gradual_increase(duration_seconds=1800)  # 30min growth
        chaos_pattern(duration_seconds=1800)   # 30min chaos
        print("=== Load scenario complete ===")
    else:
        patterns = {
            'baseline': baseline_load,
            'spike': spike_pattern,
            'periodic': periodic_pattern,
            'gradual': gradual_increase,
            'chaos': chaos_pattern
        }
        patterns[args.pattern](duration_seconds=args.duration)