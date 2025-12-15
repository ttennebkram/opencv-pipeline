#!/bin/bash
# Launch 10 parallel data generation workers
# Output to wiki_training_v3 on SamsungBlue

OUTPUT_BASE="/Volumes/SamsungBlue/ml-training/wiki_training_v3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_WORKERS=10
PAGES_PER_WORKER=10000
TRANSFORMS=10
START_ID=200000

echo "Starting $NUM_WORKERS workers at $(date)"
echo "Output: $OUTPUT_BASE"
echo "Pages per worker: $PAGES_PER_WORKER"
echo "Transforms per page: $TRANSFORMS"
echo "Total samples target: $((NUM_WORKERS * PAGES_PER_WORKER * TRANSFORMS))"
echo ""

# Create base directory
mkdir -p "$OUTPUT_BASE"

# Launch workers
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    worker_start=$((START_ID + i * PAGES_PER_WORKER * TRANSFORMS))
    worker_dir="$OUTPUT_BASE/worker_$i"
    mkdir -p "$worker_dir/images" "$worker_dir/labels"

    echo "Starting worker $i (start_id=$worker_start) -> $worker_dir"

    nice -n 10 python3 "$SCRIPT_DIR/generate_wiki_training_css3d.py" \
        --pages $PAGES_PER_WORKER \
        --transforms $TRANSFORMS \
        --start_id $worker_start \
        --output "$worker_dir" \
        > "$worker_dir/progress.log" 2>&1 &

    # Stagger starts to avoid ZIM contention
    sleep 2
done

echo ""
echo "All $NUM_WORKERS workers started."
echo "Monitor with: tail -f $OUTPUT_BASE/worker_*/progress.log"
echo "Check progress: find $OUTPUT_BASE -name '*.jpg' | wc -l"
echo ""
echo "PIDs:"
pgrep -f generate_wiki_training_css3d
