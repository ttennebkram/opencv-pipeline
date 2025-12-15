#!/bin/bash
# Sync AWS checkpoints to local disk every 3 minutes
# Run with: nohup ./sync_checkpoints.sh &

REMOTE="ubuntu@98.91.248.103"
KEY="$HOME/.ssh/ml-training.pem"
REMOTE_PATH="/opt/dlami/nvme/eighth_118k.pth"
LOCAL_DIR="/Volumes/SamsungBlue/ml-training/checkpoints_118k"
LOG_FILE="$LOCAL_DIR/sync.log"

echo "Starting checkpoint sync at $(date)" | tee -a "$LOG_FILE"
echo "Remote: $REMOTE:$REMOTE_PATH" | tee -a "$LOG_FILE"
echo "Local: $LOCAL_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

last_epoch=""

while true; do
    # Get current epoch from remote log
    epoch=$(ssh -i "$KEY" -o ConnectTimeout=10 "$REMOTE" \
        "tail -1 /opt/dlami/nvme/train_118k.log 2>/dev/null | awk '{print \$1}'" 2>/dev/null)

    # Also check continuous log
    if [ -z "$epoch" ] || [ "$epoch" = "Epoch" ]; then
        epoch=$(ssh -i "$KEY" -o ConnectTimeout=10 "$REMOTE" \
            "tail -1 /opt/dlami/nvme/train_continuous.log 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
    fi

    if [ -n "$epoch" ] && [ "$epoch" != "$last_epoch" ] && [[ "$epoch" =~ ^[0-9]+$ ]]; then
        echo "[$(date '+%H:%M:%S')] Epoch $epoch detected, syncing..." | tee -a "$LOG_FILE"

        # Download with epoch number
        scp -i "$KEY" -o ConnectTimeout=30 "$REMOTE:$REMOTE_PATH" \
            "$LOCAL_DIR/eighth_ep${epoch}.pth" 2>/dev/null

        if [ $? -eq 0 ]; then
            # Also keep a "latest" copy
            cp "$LOCAL_DIR/eighth_ep${epoch}.pth" "$LOCAL_DIR/eighth_latest.pth"

            # Get file size
            size=$(ls -lh "$LOCAL_DIR/eighth_ep${epoch}.pth" | awk '{print $5}')
            echo "[$(date '+%H:%M:%S')] Saved eighth_ep${epoch}.pth ($size)" | tee -a "$LOG_FILE"
            last_epoch="$epoch"
        else
            echo "[$(date '+%H:%M:%S')] Download failed" | tee -a "$LOG_FILE"
        fi
    fi

    sleep 180  # Check every 3 minutes
done
