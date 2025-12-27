#!/usr/bin/env python3

"""
Nexus Repository Manager Helper

This script provides utilities for interacting with Nexus repositories:
- Upload artifacts
- Sign models
- Create build metadata
- List repository contents
"""

import argparse
import hashlib
import json
import os
import requests
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class NexusClient:
    """Client for interacting with Nexus Repository Manager"""
    
    def __init__(self, nexus_url: str, username: str, password: str):
        self.nexus_url = nexus_url.rstrip('/')
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.auth = (username, password)
    
    def test_connection(self) -> bool:
        """Test connection to Nexus"""
        try:
            response = self.session.get(f"{self.nexus_url}/service/rest/v1/status", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error: Connection failed - {e}")
            return False
    
    def list_repositories(self) -> List[str]:
        """List all repositories in Nexus"""
        try:
            response = self.session.get(f"{self.nexus_url}/service/rest/v1/repositories", timeout=10)
            response.raise_for_status()
            repos = response.json()
            return [repo.get('name') for repo in repos]
        except Exception as e:
            print(f"Error: Failed to list repositories - {e}")
            return []
    
    def upload_file(self, file_path: str, repository: str, destination: Optional[str] = None) -> bool:
        """Upload a file to Nexus repository"""
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return False
        
        filename = os.path.basename(file_path)
        if destination:
            upload_path = destination
        else:
            upload_path = filename
        
        url = f"{self.nexus_url}/repository/{repository}/{upload_path}"
        
        try:
            with open(file_path, 'rb') as f:
                print(f"Uploading {filename} to {repository}...")
                response = self.session.put(url, data=f, timeout=300)
                
                if response.status_code in [200, 201, 204]:
                    print(f"✓ Successfully uploaded: {filename}")
                    print(f"  URL: {url}")
                    return True
                else:
                    print(f"✗ Upload failed (HTTP {response.status_code}): {response.text}")
                    return False
        except Exception as e:
            print(f"✗ Upload error: {e}")
            return False
    
    def list_repository_contents(self, repository: str, continuation_token: Optional[str] = None) -> Dict:
        """List contents of a repository"""
        try:
            url = f"{self.nexus_url}/service/rest/v1/components"
            params = {"repository": repository}
            if continuation_token:
                params["continuationToken"] = continuation_token
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error: Failed to list contents - {e}")
            return {}
    
    def get_file_info(self, repository: str, file_path: str) -> Optional[Dict]:
        """Get information about a file in repository"""
        try:
            url = f"{self.nexus_url}/service/rest/v1/search"
            params = {"repository": repository, "name": os.path.basename(file_path)}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json().get('items', [])
            if results:
                return results[0]
            return None
        except Exception as e:
            print(f"Error: Failed to get file info - {e}")
            return None


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Calculate hash of a file"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def sign_model(model_path: str, output_dir: str) -> bool:
    """Create checksums and metadata for a model"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found - {model_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(model_path)
    print(f"Signing model: {filename}")
    
    # Calculate hashes
    sha256_hash = calculate_file_hash(model_path, 'sha256')
    md5_hash = calculate_file_hash(model_path, 'md5')
    
    # Create signature file
    sig_file = os.path.join(output_dir, f"{filename}.sig")
    with open(sig_file, 'w') as f:
        f.write(f"SHA256: {sha256_hash}\n")
        f.write(f"MD5: {md5_hash}\n")
    
    print(f"✓ Signatures created: {sig_file}")
    
    # Create metadata
    metadata = {
        "filename": filename,
        "size": os.path.getsize(model_path),
        "timestamp": datetime.now().isoformat(),
        "hashes": {
            "sha256": sha256_hash,
            "md5": md5_hash
        }
    }
    
    metadata_file = os.path.join(output_dir, f"{filename}.metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata created: {metadata_file}")
    return True


def create_build_report(build_info: Dict, output_file: str) -> bool:
    """Create a build report"""
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "build_info": build_info,
            "artifacts": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Build report created: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to create build report: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Nexus Repository Manager Helper')
    parser.add_argument('--nexus-url', required=True, help='Nexus repository URL (e.g., http://localhost:8081)')
    parser.add_argument('--username', required=True, help='Nexus username')
    parser.add_argument('--password', required=True, help='Nexus password')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Test connection command
    subparsers.add_parser('test', help='Test connection to Nexus')
    
    # List repositories command
    subparsers.add_parser('list-repos', help='List all repositories')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload file to repository')
    upload_parser.add_argument('--file', required=True, help='File path to upload')
    upload_parser.add_argument('--repository', required=True, help='Target repository')
    upload_parser.add_argument('--destination', help='Destination path in repository')
    
    # List contents command
    list_parser = subparsers.add_parser('list', help='List repository contents')
    list_parser.add_argument('--repository', required=True, help='Repository name')
    
    # Sign model command
    sign_parser = subparsers.add_parser('sign', help='Sign a model file')
    sign_parser.add_argument('--model', required=True, help='Model file path')
    sign_parser.add_argument('--output', required=True, help='Output directory for signatures')
    
    # Build report command
    report_parser = subparsers.add_parser('report', help='Create build report')
    report_parser.add_argument('--build-id', required=True, help='Build ID')
    report_parser.add_argument('--output', required=True, help='Output file path')
    report_parser.add_argument('--artifacts', nargs='+', help='Artifact files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create Nexus client
    client = NexusClient(args.nexus_url, args.username, args.password)
    
    # Execute command
    if args.command == 'test':
        print("Testing connection to Nexus...")
        if client.test_connection():
            print("✓ Connection successful")
            return 0
        else:
            print("✗ Connection failed")
            return 1
    
    elif args.command == 'list-repos':
        repos = client.list_repositories()
        if repos:
            print(f"Found {len(repos)} repositories:")
            for repo in repos:
                print(f"  - {repo}")
            return 0
        else:
            print("No repositories found")
            return 1
    
    elif args.command == 'upload':
        if client.upload_file(args.file, args.repository, args.destination):
            return 0
        else:
            return 1
    
    elif args.command == 'list':
        contents = client.list_repository_contents(args.repository)
        if contents:
            items = contents.get('items', [])
            print(f"Repository '{args.repository}' contains {len(items)} items:")
            for item in items:
                print(f"  - {item.get('name')} ({item.get('size')} bytes)")
            return 0
        else:
            print("No contents found")
            return 1
    
    elif args.command == 'sign':
        if sign_model(args.model, args.output):
            return 0
        else:
            return 1
    
    elif args.command == 'report':
        build_info = {
            "build_id": args.build_id,
            "artifacts": args.artifacts or []
        }
        if create_build_report(build_info, args.output):
            return 0
        else:
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
