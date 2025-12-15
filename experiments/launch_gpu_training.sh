#!/bin/bash
# Launch GPU training instance and start 400-epoch run
# Instance will auto-terminate when training completes

set -e

AMI_ID="ami-063a79aa244531e14"  # 55K images
INSTANCE_TYPE="g5.xlarge"
KEY_NAME="ml-training"
SECURITY_GROUP="sg-08e1d25f98f79c51e"

echo "=== Launching GPU Training Instance ==="
echo "AMI: $AMI_ID"
echo "Instance type: $INSTANCE_TYPE"
echo ""

# Launch instance with auto-terminate on shutdown
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Public IP: $PUBLIC_IP"
echo ""
echo "Waiting 60 seconds for SSH to be ready..."
sleep 60

# SSH and start training
echo "Starting training..."
ssh -i ~/.ssh/ml-training.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'ENDSSH'
# Start training in background with nohup so it survives SSH disconnect
nohup bash -c '
LOG_FILE="/home/ubuntu/training_run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting training at $(date) ==="

cd /home/ubuntu/training/scripts
source /home/ubuntu/ml-env/bin/activate

python3 train_corners_cnn.py \
  --data /home/ubuntu/training/data/synthetic \
  --epochs 400 \
  --lr 0.002 \
  --batch_size 64 \
  --augment \
  --scheduler cosine \
  --save_checkpoints "25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400" \
  --checkpoint_dir /home/ubuntu/training/checkpoints_400ep \
  --output /home/ubuntu/training/best_model.pth \
  --csv_log /home/ubuntu/training/training_log.csv \
  --results_json /home/ubuntu/training/final_results.json

echo "=== Training complete at $(date) ==="

# Create completion marker
touch /home/ubuntu/training/TRAINING_COMPLETE

echo "=== Shutting down in 60 seconds ==="
sleep 60
sudo shutdown -h now
' &>/dev/null &

echo "Training started in background"
ENDSSH

echo ""
echo "=== Training launched! ==="
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo ""
echo "Monitor progress:"
echo "  ssh -i ~/.ssh/ml-training.pem ubuntu@$PUBLIC_IP 'tail -f ~/training_run.log'"
echo ""
echo "Check GPU usage:"
echo "  ssh -i ~/.ssh/ml-training.pem ubuntu@$PUBLIC_IP 'nvidia-smi'"
echo ""
echo "Instance will auto-terminate when training completes (~8 hours)"
echo "Estimated cost: ~\$8"
echo ""
echo "To download results after completion:"
echo "  # First, start the stopped instance or launch new one from AMI"
echo "  scp -i ~/.ssh/ml-training.pem -r ubuntu@<IP>:~/training/checkpoints_400ep ."
echo "  scp -i ~/.ssh/ml-training.pem ubuntu@<IP>:~/training/best_model.pth ."
echo "  scp -i ~/.ssh/ml-training.pem ubuntu@<IP>:~/training/training_log.csv ."
