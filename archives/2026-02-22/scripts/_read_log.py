import subprocess
lines = open("/root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/console.log").readlines()
for i, line in enumerate(lines):
    print(f"{i:03d}: {line}", end="")
