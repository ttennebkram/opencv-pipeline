#!/usr/bin/env python3
"""
collect_results.py

Collect training results from multiple EC2 instances and merge them.

Usage:
    # Collect from all running instances
    python3 collect_results.py --key ~/.ssh/ml-training.pem

    # Collect from specific IPs
    python3 collect_results.py --key ~/.ssh/ml-training.pem --ips 1.2.3.4 5.6.7.8
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import csv

def get_running_instances():
    """Get IPs of all running EC2 instances with our security group."""
    try:
        result = subprocess.run([
            'aws', 'ec2', 'describe-instances',
            '--filters',
            'Name=instance-state-name,Values=running',
            'Name=instance.group-id,Values=sg-08e1d25f98f79c51e',
            '--query', 'Reservations[*].Instances[*].PublicIpAddress',
            '--output', 'text'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"Error getting instances: {result.stderr}")
            return []

        ips = result.stdout.strip().split()
        return [ip for ip in ips if ip and ip != 'None']
    except Exception as e:
        print(f"Error: {e}")
        return []

def collect_from_instance(ip, key_path, results_dir):
    """SSH to instance, collect results JSON, save locally."""
    print(f"  Collecting from {ip}...")

    try:
        # Get the results file
        result = subprocess.run([
            'ssh', '-i', key_path,
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=10',
            f'ubuntu@{ip}',
            'cat ~/training/results.json 2>/dev/null || echo "{}"'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"    SSH failed: {result.stderr}")
            return None

        data = json.loads(result.stdout.strip())
        if not data:
            print(f"    No results found")
            return None

        # Add collection metadata
        data['collected_at'] = datetime.now().isoformat()
        data['source_ip'] = ip

        # Save individual result
        instance_id = data.get('instance_id', ip.replace('.', '_'))
        result_file = results_dir / f"result_{instance_id}.json"
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"    OK: pixel_error={data.get('best_pixel_error', 'N/A')}")
        return data

    except json.JSONDecodeError as e:
        print(f"    Invalid JSON: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    Timeout")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

def merge_results(all_results, output_file):
    """Merge all results into a summary CSV."""
    if not all_results:
        print("No results to merge")
        return

    # Get all unique keys across results
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())

    # Sort keys for consistent output
    fieldnames = sorted(all_keys)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\nSaved merged results to {output_file}")

    # Print summary statistics
    print("\n=== Summary ===")
    pixel_errors = [r['best_pixel_error'] for r in all_results if 'best_pixel_error' in r]
    if pixel_errors:
        print(f"Results collected: {len(all_results)}")
        print(f"Best pixel error: {min(pixel_errors):.1f}")
        print(f"Worst pixel error: {max(pixel_errors):.1f}")
        print(f"Mean pixel error: {sum(pixel_errors)/len(pixel_errors):.1f}")

        # Find best result
        best = min(all_results, key=lambda x: x.get('best_pixel_error', float('inf')))
        print(f"\nBest result:")
        print(f"  Instance: {best.get('instance_id', 'unknown')}")
        print(f"  Seed: {best.get('seed', 'unknown')}")
        print(f"  LR: {best.get('lr', 'unknown')}")
        print(f"  Pixel error: {best.get('best_pixel_error', 'unknown')}")

def main():
    parser = argparse.ArgumentParser(description="Collect results from EC2 instances")
    parser.add_argument('--key', type=str, default='~/.ssh/ml-training.pem',
                        help='SSH key path')
    parser.add_argument('--ips', nargs='*', help='Specific IPs to collect from')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    key_path = os.path.expanduser(args.key)
    results_dir = Path(args.output)
    results_dir.mkdir(exist_ok=True)

    # Get instance IPs
    if args.ips:
        ips = args.ips
    else:
        print("Discovering running instances...")
        ips = get_running_instances()

    if not ips:
        print("No instances found")
        return 1

    print(f"Found {len(ips)} instance(s)")

    # Collect from each
    all_results = []
    for ip in ips:
        result = collect_from_instance(ip, key_path, results_dir)
        if result:
            all_results.append(result)

    # Merge results
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        merge_results(all_results, results_dir / f"summary_{timestamp}.csv")
        merge_results(all_results, results_dir / "summary_latest.csv")

    return 0

if __name__ == "__main__":
    sys.exit(main())
