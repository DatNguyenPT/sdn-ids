#!/usr/bin/env python3
"""
MLflow Health Check Script

This script checks if MLflow server is healthy and accessible.

Usage:
    python scripts/check_mlflow_health.py --mlflow-url http://localhost:5002
"""

import argparse
import requests
import sys
import time
from typing import Optional


def check_mlflow_health(mlflow_url: str, timeout: int = 30) -> bool:
    """Check if MLflow server is healthy"""
    try:
        # Remove trailing slash
        mlflow_url = mlflow_url.rstrip('/')
        
        # Try to access MLflow UI (basic connectivity check)
        print(f"Checking MLflow server at {mlflow_url}...")
        
        response = requests.get(
            f"{mlflow_url}/",
            timeout=timeout,
            allow_redirects=True
        )
        
        if response.status_code in [200, 302]:
            print(f"✅ MLflow server is accessible (HTTP {response.status_code})")
            return True
        else:
            print(f"⚠️  MLflow server returned HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to MLflow server at {mlflow_url}")
        print("   Make sure MLflow server is running and accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout connecting to MLflow server (>{timeout}s)")
        return False
    except Exception as e:
        print(f"❌ Error checking MLflow health: {e}")
        return False


def check_mlflow_api(mlflow_url: str, timeout: int = 30) -> bool:
    """Check if MLflow REST API is working"""
    try:
        mlflow_url = mlflow_url.rstrip('/')
        
        # Check experiments API endpoint
        print("Checking MLflow REST API...")
        
        response = requests.get(
            f"{mlflow_url}/api/2.0/mlflow/experiments/search",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            experiments = data.get('experiments', [])
            print(f"✅ MLflow REST API is working (found {len(experiments)} experiment(s))")
            return True
        else:
            print(f"⚠️  MLflow REST API returned HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  Error checking MLflow API: {e}")
        return False


def wait_for_mlflow(mlflow_url: str, max_wait: int = 60, check_interval: int = 2) -> bool:
    """Wait for MLflow server to become available"""
    print(f"Waiting for MLflow server to be ready (max {max_wait}s)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_mlflow_health(mlflow_url, timeout=5):
            return True
        print(f"   Retrying in {check_interval}s...")
        time.sleep(check_interval)
    
    return False


def main():
    parser = argparse.ArgumentParser(description='MLflow Health Check')
    parser.add_argument(
        '--mlflow-url',
        type=str,
        default='http://localhost:5002',
        help='MLflow server URL (default: http://localhost:5002)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for MLflow server to become available'
    )
    parser.add_argument(
        '--max-wait',
        type=int,
        default=60,
        help='Maximum time to wait in seconds (default: 60)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLflow Health Check")
    print("=" * 60)
    print(f"MLflow URL: {args.mlflow_url}")
    print("=" * 60)
    
    # Wait for MLflow if requested
    if args.wait:
        if not wait_for_mlflow(args.mlflow_url, max_wait=args.max_wait):
            print("\n❌ MLflow server did not become available")
            sys.exit(1)
    else:
        # Check health immediately
        if not check_mlflow_health(args.mlflow_url, timeout=args.timeout):
            print("\n❌ MLflow health check failed")
            sys.exit(1)
    
    # Check API
    api_ok = check_mlflow_api(args.mlflow_url, timeout=args.timeout)
    
    # Summary
    print("\n" + "=" * 60)
    print("Health Check Summary")
    print("=" * 60)
    print("✅ MLflow server is accessible")
    if api_ok:
        print("✅ MLflow REST API is working")
    else:
        print("⚠️  MLflow REST API check failed (may still be OK)")
    print("=" * 60)
    
    if api_ok:
        print("\n✅ MLflow health check passed!")
        sys.exit(0)
    else:
        print("\n⚠️  MLflow health check completed with warnings")
        sys.exit(0)  # Don't fail if API check fails, server might still work


if __name__ == "__main__":
    main()

