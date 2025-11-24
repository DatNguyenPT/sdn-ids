#!/usr/bin/env python3
"""
Flower Server for Federated Learning

This script starts a Flower server that receives model weights from clients.
The server aggregates weights from multiple clients using Federated Averaging (FedAvg).

Usage:
    python flower_server.py
    python flower_server.py --address 0.0.0.0 --port 8080
"""

import argparse
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Flower Federated Learning Server')
    parser.add_argument(
        '--address',
        type=str,
        default='0.0.0.0',
        help='Server address (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Server port (default: 8080)'
    )
    parser.add_argument(
        '--min-clients',
        type=int,
        default=1,
        help='Minimum number of clients required (default: 1)'
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=1,
        help='Number of federated learning rounds (default: 1)'
    )
    parser.add_argument(
        '--fraction-fit',
        type=float,
        default=1.0,
        help='Fraction of clients used for training (default: 1.0)'
    )
    parser.add_argument(
        '--fraction-evaluate',
        type=float,
        default=1.0,
        help='Fraction of clients used for evaluation (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Define federated learning strategy
    # FedAvg: Federated Averaging - averages model weights from clients
    strategy = FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    
    # Server configuration
    config = ServerConfig(num_rounds=args.num_rounds)
    
    print("\n" + "="*60)
    print(" Flower Federated Learning Server")
    print("="*60)
    print(f" Address:        {args.address}")
    print(f" Port:           {args.port}")
    print(f" Min Clients:    {args.min_clients}")
    print(f" Rounds:         {args.num_rounds}")
    print(f" Fraction Fit:   {args.fraction_fit}")
    print(f" Fraction Eval:  {args.fraction_evaluate}")
    print("="*60)
    print("\n Waiting for clients to connect...")
    print(" Press Ctrl+C to stop the server\n")
    
    try:
        # Start Flower server
        fl.server.start_server(
            server_address=f"{args.address}:{args.port}",
            config=config,
            strategy=strategy,
        )
    except KeyboardInterrupt:
        logger.info("\n Server stopped by user")
    except Exception as e:
        logger.error(f" Server error: {e}")
        raise


if __name__ == "__main__":
    main()

