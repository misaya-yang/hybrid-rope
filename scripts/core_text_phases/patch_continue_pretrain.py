#!/usr/bin/env python
"""Patch run_gqa_evq_experiment.py to support continued pretraining.

Adds two arguments:
  --init_from   PATH   Load model weights from this .pt file before training
                       (used for continued pretraining from a saved checkpoint)
  --continue_lr FLOAT  Override initial LR (default 1e-5 for continued pretrain)
  --no_intermediate_ckpts  Skip 50%/75% checkpoint saves (just save final model.pt)

Usage:
  # Run this patch once on the server:
  python patch_continue_pretrain.py

  # Then run continued pretraining:
  python run_gqa_evq_experiment.py \
      --init_from /path/to/model.pt \
      --continue_lr 1e-5 \
      --no_intermediate_ckpts \
      ...other args...
"""

path = "/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py"
with open(path) as f:
    code = f.read()

# 1. Add --init_from, --continue_lr, --no_intermediate_ckpts arguments
old_compile_arg = '    parser.add_argument("--compile", action="store_true",'
new_args = '''    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to model.pt to load as starting weights (continued pretraining)")
    parser.add_argument("--continue_lr", type=float, default=None,
                        help="Override initial LR for continued pretraining (e.g. 1e-5)")
    parser.add_argument("--no_intermediate_ckpts", action="store_true",
                        help="Skip 50%/75% checkpoint saves; only save final model.pt")
    parser.add_argument("--compile", action="store_true",'''
assert old_compile_arg in code, "--compile arg not found!"
code = code.replace(old_compile_arg, new_args)

# 2. Inject --continue_lr into cfg before train_model is called
# Insert after the GQA inject block, near where cfg is finalized
old_cfg_finalize = '    cfg["passkey_mix_ratio"] = ('
new_cfg_finalize = '''    # Continued pretraining LR override
    if args.continue_lr is not None:
        cfg["lr"] = args.continue_lr
        print(f"  [continue] Overriding LR to {args.continue_lr:.2e} for continued pretraining")

    # No intermediate checkpoints flag
    if args.no_intermediate_ckpts:
        cfg["no_intermediate_ckpts"] = True
        print("  [continue] Will skip 50%/75% intermediate checkpoints")

    cfg["passkey_mix_ratio"] = ('''
assert old_cfg_finalize in code, "cfg passkey_mix_ratio line not found!"
code = code.replace(old_cfg_finalize, new_cfg_finalize)

# 3. Pass init_from and no_intermediate_ckpts to run_single via cfg
# Find the run_single call and add init_from
old_run_single_call = '''        result, passkey_res, pi_res = run_single(
                    tau=tau,
                    seed=seed,
                    cfg=cfg,'''
new_run_single_call = '''        # Inject init_from path into cfg for run_single to pick up
        if args.init_from:
            cfg["init_from"] = args.init_from
        result, passkey_res, pi_res = run_single(
                    tau=tau,
                    seed=seed,
                    cfg=cfg,'''
if old_run_single_call in code:
    code = code.replace(old_run_single_call, new_run_single_call)
    print("  Patched run_single call")
else:
    # Try alternative
    old_run_single_call2 = "        result, passkey_res, pi_res = run_single("
    idx = code.find(old_run_single_call2)
    if idx >= 0:
        code = code[:idx] + '''        if args.init_from:
            cfg["init_from"] = args.init_from
        ''' + code[idx:]
        print("  Patched run_single call (alternative)")
    else:
        print("  WARNING: run_single call not found, manual patch needed")

# 4. In run_single: load init_from weights after model creation
old_model_create = '''    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)
    save_model = model
    if cfg.get("compile", False):'''
new_model_create = '''    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)
    # Load from pretrained checkpoint if specified
    init_from = cfg.get("init_from", None)
    if init_from:
        print(f"  [init_from] Loading weights from {init_from}")
        sd = torch.load(init_from, map_location=DEVICE, weights_only=True)
        clean = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
        if missing:
            print(f"  [init_from] WARNING: missing keys: {missing[:5]}")
        if unexpected:
            print(f"  [init_from] WARNING: unexpected keys: {unexpected[:5]}")
        print(f"  [init_from] Loaded successfully")
    save_model = model
    if cfg.get("compile", False):'''
assert old_model_create in code, "model creation block not found!"
code = code.replace(old_model_create, new_model_create)

# 5. In train_model: respect no_intermediate_ckpts flag
old_save_steps = '''    save_steps = {
        max(1, math.ceil(steps * 0.50)): "50",
        max(1, math.ceil(steps * 0.75)): "75",
        steps: "100",
    }'''
new_save_steps = '''    if cfg.get("no_intermediate_ckpts", False):
        save_steps = {steps: "100"}
    else:
        save_steps = {
            max(1, math.ceil(steps * 0.50)): "50",
            max(1, math.ceil(steps * 0.75)): "75",
            steps: "100",
        }'''
assert old_save_steps in code, "save_steps dict not found!"
code = code.replace(old_save_steps, new_save_steps)

with open(path, "w") as f:
    f.write(code)

# Verify
with open(path) as f:
    content = f.read()
checks = ["--init_from", "--continue_lr", "--no_intermediate_ckpts",
          "init_from", "continue_lr", "no_intermediate_ckpts", "init_from] Loading"]
for c in checks:
    count = content.count(c)
    status = "OK" if count > 0 else "MISSING"
    print(f"  {status}: {c} ({count} occurrences)")

print("\ncontinued pretraining patch applied successfully!")
