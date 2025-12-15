#!/bin/bash
# Continuous S3 sync for new training data
# Run with: nohup ./sync_to_s3.sh &

LOCAL_DIR="/Volumes/SamsungBlue/ml-training/wiki_training_v3"
S3_BUCKET="s3://ml-training-wiki-homography/wiki_training_v3/"
LOG_FILE="$LOCAL_DIR/s3_sync.log"

echo "Starting S3 sync at $(date)" | tee "$LOG_FILE"
echo "Local: $LOCAL_DIR" | tee -a "$LOG_FILE"
echo "S3: $S3_BUCKET" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    count=$(find "$LOCAL_DIR" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] Local count: $count samples - syncing..." | tee -a "$LOG_FILE"

    aws s3 sync "$LOCAL_DIR" "$S3_BUCKET" \
        --exclude "*.log" \
        --exclude ".DS_Store" \
        --quiet \
        2>&1 | tee -a "$LOG_FILE"

    s3_count=$(aws s3 ls "$S3_BUCKET" --recursive | grep -c "\.jpg$" || echo "0")
    echo "[$(date '+%H:%M:%S')] S3 count: $s3_count samples" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    sleep 300  # Sync every 5 minutes
done
