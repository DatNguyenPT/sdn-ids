#!/usr/bin/env python3
"""
Wait for FL Training Completion

Monitors FL training and waits for completion or timeout.

Usage:
    python mlops/wait_for_fl_completion.py --model-type MLPv2 --max-wait 3600
"""

import argparse
import time
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_training_status(model_type: str) -> bool:
    """Check if FL training has completed"""
    try:
        # Check server logs for completion message
        result = subprocess.run(
            ['docker', 'compose', 'logs', f'flower-server-{model_type.lower()}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        logs = result.stdout + result.stderr
        
        # Check for completion indicators
        completion_indicators = [
            'Training finished',
            'All.*rounds completed',
            '✅ All',
            'Training complete'
        ]
        
        for indicator in completion_indicators:
            if indicator.lower() in logs.lower():
                logger.info(f"Training completion detected: {indicator}")
                return True
        
        return False
    except Exception as e:
        logger.debug(f"Error checking status: {e}")
        return False


def wait_for_completion(model_type: str, max_wait: int, check_interval: int = 10):
    """
    Wait for FL training to complete
    
    Args:
        model_type: Model type being trained
        max_wait: Maximum time to wait (seconds)
        check_interval: Interval between checks (seconds)
    """
    logger.info(f"Waiting for FL training completion (max {max_wait}s)...")
    
    start_time = time.time()
    elapsed = 0
    
    while elapsed < max_wait:
        if check_training_status(model_type):
            logger.info("✅ FL training completed successfully!")
            return True
        
        elapsed = time.time() - start_time
        remaining = max_wait - elapsed
        
        if remaining > 0:
            logger.info(f"Training in progress... ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)")
            time.sleep(check_interval)
        else:
            break
    
    logger.warning(f"⏱️ Timeout reached ({max_wait}s). Training may still be running.")
    return False


def main():
    parser = argparse.ArgumentParser(description='Wait for FL training completion')
    parser.add_argument('--model-type', type=str, required=True,
                       help='Model type (MLPv2, LSTM, CNN1D, CNN_LSTM)')
    parser.add_argument('--max-wait', type=int, default=3600,
                       help='Maximum wait time in seconds (default: 3600)')
    parser.add_argument('--check-interval', type=int, default=10,
                       help='Check interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    success = wait_for_completion(
        model_type=args.model_type,
        max_wait=args.max_wait,
        check_interval=args.check_interval
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

