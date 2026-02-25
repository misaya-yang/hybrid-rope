#!/bin/bash
# Monitor experiment progress

echo "=== Phase 1 Experiment Monitor ==="
echo ""

# Check if process is running
PID=$(pgrep -f "phase1_prior_softmax_v2/experiment.py" | head -1)
if [ -n "$PID" ]; then
    echo "✓ Experiment is running (PID: $PID)"
    echo "  CPU: $(ps -p $PID -o %cpu= 2>/dev/null || echo 'N/A')%"
    echo "  MEM: $(ps -p $PID -o %mem= 2>/dev/null || echo 'N/A')%"
else
    echo "✗ Experiment is not running"
fi

echo ""
echo "=== Latest Progress ==="
if [ -f "experiment.log" ]; then
    echo "Log file size: $(ls -lh experiment.log | awk '{print $5}')"
    echo ""
    echo "Last 20 lines:"
    tail -20 experiment.log
else
    echo "No log file found"
fi

echo ""
echo "=== Results ==="
if [ -d "results" ]; then
    ls -lh results/ 2>/dev/null || echo "No results yet"
else
    echo "No results directory"
fi
