#!/bin/bash
#
# launch_timing_test.sh
#
# Launch 10 EC2 instances from the training AMI and run 10 epochs each
# to time how long training takes.
#
# Usage:
#   ./launch_timing_test.sh
#

set -e

# Configuration from our setup
AMI_ID="ami-0dd2d28e9f738aa69"  # With 10k training data + scipy baked in
SECURITY_GROUP="sg-08e1d25f98f79c51e"
KEY_NAME="ml-training"
KEY_PATH="$HOME/.ssh/ml-training.pem"
INSTANCE_TYPE="t3.medium"  # CPU instance for timing test (cheap)
COUNT=10

echo "=== EC2 Multi-Instance Timing Test ==="
echo "AMI: $AMI_ID"
echo "Instance Type: $INSTANCE_TYPE"
echo "Count: $COUNT"
echo "Epochs per instance: 10"
echo ""

# Launch instances
echo "Launching $COUNT instances..."
INSTANCE_IDS=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --count "$COUNT" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ml-timing-test}]" \
    --query 'Instances[*].InstanceId' \
    --output text)

echo "Instance IDs: $INSTANCE_IDS"
echo ""

# Wait for instances to be running
echo "Waiting for instances to enter running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_IDS
echo "All instances running."
echo ""

# Get public IPs
echo "Getting public IPs..."
IPS=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_IDS \
    --query 'Reservations[*].Instances[*].PublicIpAddress' \
    --output text)

echo "IPs: $IPS"
echo ""

# Save IPs to file for later collection
echo "$IPS" | tr '\t' '\n' > /tmp/timing_test_ips.txt
echo "Saved IPs to /tmp/timing_test_ips.txt"
echo ""

# Wait a bit for SSH to be available
echo "Waiting 60 seconds for SSH to be available..."
sleep 60

# Start training on each instance
echo "Starting training on each instance (10 epochs, unique seeds)..."
for IP in $IPS; do
    echo "  Starting on $IP..."
    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "ubuntu@$IP" << 'TRAINING_EOF' &
source ~/ml-env/bin/activate
cd ~/training/scripts

# Run 10-epoch timing test with auto-generated seed
# Results go to ~/training/results.json
python3 train_corners_cnn.py \
    --epochs 10 \
    --lr 0.002 \
    --batch_size 64 \
    --augment \
    --scheduler cosine_warm \
    --data ~/training/data/synthetic \
    --results_json ~/training/results.json \
    --instance_id "$(hostname)" \
    2>&1 | tee ~/training/training.log
TRAINING_EOF
    sleep 2  # Stagger starts slightly
done

echo ""
echo "=== Training started on all $COUNT instances ==="
echo ""
echo "Monitor progress with:"
echo "  for ip in \$(cat /tmp/timing_test_ips.txt); do echo \"=== \$ip ===\"; ssh -i $KEY_PATH ubuntu@\$ip 'tail -3 ~/training/training.log 2>/dev/null || echo waiting...'; done"
echo ""
echo "Collect results when done with:"
echo "  python3 collect_results.py --key $KEY_PATH --ips \$(cat /tmp/timing_test_ips.txt | tr '\n' ' ')"
echo ""
echo "Terminate instances when done with:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_IDS"
