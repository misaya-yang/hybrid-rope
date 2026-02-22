@echo off
set PYTHON_EXE=E:\Anaconda\envs\aidemo\python.exe
set SCRIPT_DIR=E:\rope\hybrid-rope\2026-02-22\scripts
set DATA_DIR=E:\rope\hybrid-rope\data\phase_transition
set OUT_DIR=E:\rope\hybrid-rope\results\phase_transition

echo =======================================================
echo          STARTING PHASE TRANSITION EXPERIMENT
echo =======================================================
echo.
echo Running on 4070 Super (12GB VRAM Profile)
echo Target steps: 300 per configuration
echo.

echo --- [1/4] Running Standard RoPE on Continuous Text (Gamma=1.0) ---
%PYTHON_EXE% %SCRIPT_DIR%\train_phase_transition.py --data_path %DATA_DIR%\gamma_1.00_seq1024 --rope_type standard --max_steps 300 --output_dir %OUT_DIR%\gamma_1.0

echo.
echo --- [2/4] Running Hybrid RoPE on Continuous Text (Gamma=1.0) ---
%PYTHON_EXE% %SCRIPT_DIR%\train_phase_transition.py --data_path %DATA_DIR%\gamma_1.00_seq1024 --rope_type hybrid --max_steps 300 --output_dir %OUT_DIR%\gamma_1.0

echo.
echo --- [3/4] Running Standard RoPE on Fragmented Text (Gamma=0.0) ---
%PYTHON_EXE% %SCRIPT_DIR%\train_phase_transition.py --data_path %DATA_DIR%\gamma_0.00_seq1024 --rope_type standard --max_steps 300 --output_dir %OUT_DIR%\gamma_0.0

echo.
echo --- [4/4] Running Hybrid RoPE on Fragmented Text (Gamma=0.0) ---
%PYTHON_EXE% %SCRIPT_DIR%\train_phase_transition.py --data_path %DATA_DIR%\gamma_0.00_seq1024 --rope_type hybrid --max_steps 300 --output_dir %OUT_DIR%\gamma_0.0

echo.
echo =======================================================
echo             EXPERIMENT FULLY COMPLETED!
echo =======================================================
pause
