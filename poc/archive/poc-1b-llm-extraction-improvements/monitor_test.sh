#!/bin/bash
# Monitor test progress

LOG_FILE="test_run.log"
LOG_DETAILED="artifacts/logs/sentence_level_*.log"

echo "==================================================================="
echo "POC-1b Sentence-Level Extraction Test Monitor"
echo "==================================================================="
echo ""

# Check if process is running
if ps aux | grep -q "[t]est_sentence_level_extraction.py"; then
    echo "✓ Test is RUNNING (PID: $(pgrep -f test_sentence_level_extraction.py))"
else
    echo "✗ Test is NOT RUNNING"
fi

echo ""
echo "-------------------------------------------------------------------"
echo "Progress (from test_run.log)"
echo "-------------------------------------------------------------------"

# Count completed chunks
COMPLETED=$(grep -c "METRIC | f1=" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Chunks completed: $COMPLETED / 10"

# Show last chunk result
echo ""
echo "Last chunk metrics:"
tail -100 "$LOG_FILE" | grep -A 5 "METRIC | precision" | tail -6

# Show Sonnet filtering decisions from last chunk
echo ""
echo "-------------------------------------------------------------------"
echo "Latest Sonnet filtering decisions:"
echo "-------------------------------------------------------------------"
tail -200 "$LOG_FILE" | grep -E "(KEEP|REMOVE)" | tail -20

# Show timing
echo ""
echo "-------------------------------------------------------------------"
echo "Timing:"
echo "-------------------------------------------------------------------"
START_TIME=$(head -20 "$LOG_FILE" | grep -oP '\[\K[0-9:]+' | head -1)
CURRENT_TIME=$(tail -20 "$LOG_FILE" | grep -oP '\[\K[0-9:]+' | tail -1)
echo "Started: $START_TIME"
echo "Latest: $CURRENT_TIME"

# Calculate chunks per minute
if [ "$COMPLETED" -gt 0 ]; then
    ELAPSED_MINS=$(tail -100 "$LOG_FILE" | grep "METRIC | time=" | awk -F"=" '{sum+=$2} END {print sum/60}')
    echo "Elapsed: ~${ELAPSED_MINS} minutes"
    RATE=$(echo "scale=1; $COMPLETED / $ELAPSED_MINS" | bc 2>/dev/null || echo "N/A")
    echo "Rate: ~${RATE} chunks/min"
    REMAINING=$(echo "scale=1; (10 - $COMPLETED) / $RATE" | bc 2>/dev/null || echo "N/A")
    echo "Estimated remaining: ~${REMAINING} minutes"
fi

echo ""
echo "==================================================================="
echo "Use: tail -f test_run.log  # to watch live"
echo "==================================================================="
