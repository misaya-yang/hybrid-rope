#!/bin/bash
while true; do
    echo "=== \$(date) ===" >> /opt/dfrope/monitor.log
    echo "GPU: \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> /opt/dfrope/monitor.log
    echo "CPU: \$(top -bn1 | grep 'Cpu(s)')" >> /opt/dfrope/monitor.log
    tail -5 /opt/dfrope/results/350m_validation/train.log >> /opt/dfrope/monitor.log
    echo "---" >> /opt/dfrope/monitor.log
    sleep 60
done
