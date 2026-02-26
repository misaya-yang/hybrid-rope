# Attention Audit Report (Read-Only)

- Generated at: `2026-02-26 10:17:32`
- Target: `root@connect.bjb1.seetacloud.com:52592`

## Running Python Scripts

### PID `910`
- Highlighted: `no`
- PPID: `888`
- Elapsed (s): `125972`
- Working dir: `/root`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/bin/jupyter-lab --allow-root --config=/init/jupyter/jupyter_config.py`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97597`
- Highlighted: `yes`
- PPID: `56073`
- Elapsed (s): `10206`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python scripts/eval_longbench.py --base_model_path /root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct --adapter_path /root/autodl-tmp/dfrope/hybrid-rope/artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_anchored_sigmoid_1337 --model_alias anchored_sigmoid --skip_base_unfinetuned --variant custom --custom_inv_freq_path /root/autodl-tmp/dfrope/hybrid-rope/artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_anchored_sigmoid_1337/artifacts/custom_inv_freq.pt --task_set lb21 --max_samples_per_task 0 --max_input_tokens 16384 --prompt_source official --chat_template auto --truncate_mode middle --max_new_tokens_policy official --score_scale pct --strict_parity_check --manifest_json /root/autodl-tmp/dfrope/hybrid-rope/artifacts/manifests/longbench_manifest_qwen_ctx16384_seed1337.json --save_per_sample_traces 1 --trace_output_max_chars 1024 --repro_manifest_dir /root/autodl-tmp/dfrope/hybrid-rope/artifacts/repro_manifest/h2_qwen_fast400_seed1337_anchored_sigmoid --seed 1337 --output_json /root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json`
- Output JSON: `/root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json`
- Output dir: ``
- Dataset hint: ``
- Log file: `/root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/eval_longbench.log`
- Last 30 log lines:
```text
2026-02-26 10:15:43 | INFO | task=passage_retrieval_en progress=70/200 running_retrieval_en_raw=0.9571
2026-02-26 10:15:52 | INFO | task=passage_retrieval_en progress=80/200 running_retrieval_en_raw=0.9625
2026-02-26 10:16:02 | INFO | task=passage_retrieval_en progress=90/200 running_retrieval_en_raw=0.9556
2026-02-26 10:16:11 | INFO | task=passage_retrieval_en progress=100/200 running_retrieval_en_raw=0.9600
2026-02-26 10:16:20 | INFO | task=passage_retrieval_en progress=110/200 running_retrieval_en_raw=0.9636
2026-02-26 10:16:29 | INFO | task=passage_retrieval_en progress=120/200 running_retrieval_en_raw=0.9667
2026-02-26 10:16:39 | INFO | task=passage_retrieval_en progress=130/200 running_retrieval_en_raw=0.9692
2026-02-26 10:16:48 | INFO | task=passage_retrieval_en progress=140/200 running_retrieval_en_raw=0.9714
2026-02-26 10:16:57 | INFO | task=passage_retrieval_en progress=150/200 running_retrieval_en_raw=0.9733
2026-02-26 10:17:06 | INFO | task=passage_retrieval_en progress=160/200 running_retrieval_en_raw=0.9750
2026-02-26 10:17:15 | INFO | task=passage_retrieval_en progress=170/200 running_retrieval_en_raw=0.9765
2026-02-26 10:17:24 | INFO | task=passage_retrieval_en progress=180/200 running_retrieval_en_raw=0.9778
2026-02-26 10:17:33 | INFO | task=passage_retrieval_en progress=190/200 running_retrieval_en_raw=0.9737
2026-02-26 10:17:42 | INFO | task=passage_retrieval_en progress=200/200 running_retrieval_en_raw=0.9750
2026-02-26 10:17:45 | INFO | task=passage_retrieval_zh progress=10/200 running_retrieval_zh_raw=1.0000
2026-02-26 10:17:49 | INFO | task=passage_retrieval_zh progress=20/200 running_retrieval_zh_raw=0.9500
2026-02-26 10:17:52 | INFO | task=passage_retrieval_zh progress=30/200 running_retrieval_zh_raw=0.9333
2026-02-26 10:17:55 | INFO | task=passage_retrieval_zh progress=40/200 running_retrieval_zh_raw=0.9250
2026-02-26 10:17:58 | INFO | task=passage_retrieval_zh progress=50/200 running_retrieval_zh_raw=0.9200
2026-02-26 10:18:02 | INFO | task=passage_retrieval_zh progress=60/200 running_retrieval_zh_raw=0.9333
2026-02-26 10:18:05 | INFO | task=passage_retrieval_zh progress=70/200 running_retrieval_zh_raw=0.9429
2026-02-26 10:18:09 | INFO | task=passage_retrieval_zh progress=80/200 running_retrieval_zh_raw=0.9500
2026-02-26 10:18:12 | INFO | task=passage_retrieval_zh progress=90/200 running_retrieval_zh_raw=0.9444
2026-02-26 10:18:15 | INFO | task=passage_retrieval_zh progress=100/200 running_retrieval_zh_raw=0.9300
2026-02-26 10:18:19 | INFO | task=passage_retrieval_zh progress=110/200 running_retrieval_zh_raw=0.9364
2026-02-26 10:18:22 | INFO | task=passage_retrieval_zh progress=120/200 running_retrieval_zh_raw=0.9417
2026-02-26 10:18:25 | INFO | task=passage_retrieval_zh progress=130/200 running_retrieval_zh_raw=0.9308
2026-02-26 10:18:29 | INFO | task=passage_retrieval_zh progress=140/200 running_retrieval_zh_raw=0.9286
2026-02-26 10:18:32 | INFO | task=passage_retrieval_zh progress=150/200 running_retrieval_zh_raw=0.9333
2026-02-26 10:18:35 | INFO | task=passage_retrieval_zh progress=160/200 running_retrieval_zh_raw=0.9375
```

### PID `97640`
- Highlighted: `no`
- PPID: `97597`
- Elapsed (s): `10198`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97662`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97664`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97665`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97667`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97669`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97671`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97673`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97675`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97677`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97679`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97681`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97683`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97685`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97687`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97689`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97692`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97694`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97695`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97697`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97699`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97701`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

### PID `97703`
- Highlighted: `no`
- PPID: `97640`
- Elapsed (s): `10196`
- Working dir: `/root/autodl-tmp/dfrope/hybrid-rope`
- Script cmd: `/root/miniconda3/bin/python /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=22 --parent=97597 --read-fd=45 --write-fd=48 --torch-key=H4QweOk00IXhifNSBFt33pU0hqEJ8U7VJTQvH3venIU=`
- Output JSON: ``
- Output dir: ``
- Dataset hint: ``
- Log file: ``
- Last 30 log lines:
```text
(no readable log tail)
```

