#!/bin/bash
# Monitor Phase 1 experiment progress

LOG_FILE="/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict/phase1_prior_softmax/experiment.log"

echo "========================================"
echo "Phase 1 Experiment Monitor"
echo "========================================"
echo ""

# Check if running
PID=$(pgrep -f "experiment.py" | head -1)
if [ -n "$PID" ]; then
    echo "✓ Experiment running (PID: $PID)"
    echo "  Uptime: $(ps -o etime= -p $PID)"
    echo "  CPU: $(ps -o %cpu= -p $PID)%"
    echo "  MEM: $(ps -o %mem= -p $PID)%"
else
    echo "✗ Experiment not running"
fi

echo ""
echo "========================================"
echo "Latest Progress"
echo "========================================"

if [ -f "$LOG_FILE" ]; then
    # Get last evaluation results
    tail -100 "$LOG_FILE" | grep -E "(Step|Val PPL|Group|Training)" | tail -20
    
    echo ""
    echo "========================================"
    echo "Summary Stats"
    echo "========================================"
    
    # Count completed groups
    GROUPS=$(grep -c "Training Group:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "Groups started: $GROUPS/7"
    
    # Check for results
    RESULTS_DIR="/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict/results"
    if [ -d "$RESULTS_DIR" ]; then
        CSV_COUNT=$(ls -1 "$RESULTS_DIR"/*.csv 2>/dev/null | wc -l)
        if [ "$CSV_COUNT" -gt 0 ]; then
            echo ""
            echo "✓ Results available:"
            ls -lh "$RESULTS_DIR"/*.csv | tail -3
        fi
    fi
else
    echo "Log file not found yet..."
fi

echo ""
echo "========================================"
echo "Monitor Commands"
echo "========================================"
echo "Watch live:     tail -f $LOG_FILE"
echo "Check status:   $0"
echo "Kill process:   pkill -f experiment.py"
