# Extended target-free allocation M4 matrix (2026-08-24)

- Status: **COMPLETE_INTERNAL_PRELIMINARY**
- Canonical verdict: **SCREEN_UNRESOLVED**
- Scope: internal preliminary research only; no manuscript claim.

## Stage summaries

### stageA_base256_50m (base=256, tokens/arm=50,331,648)

This is the 50M/base-256 budget matrix. Its weighted OOD tail NLLs (lower is
better) were FMRoPE 6.714540, anchored EVQ-Cosh 6.894834, phase-isotropy
6.779985, pair-volume 6.811190, and min-eigenvalue 6.862728. Paired
tail-NLL differences (arm minus FMRoPE; negative is better), in the order
256/512/1024/2048, were:

| Arm | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| anchored EVQ-Cosh | +0.044959 | +0.023476 | +0.259081 | +0.200992 |
| phase-isotropy | +0.029492 | +0.073365 | −0.035179 | +0.135190 |
| pair-volume | −0.028543 | +0.017985 | +0.069266 | +0.156040 |
| min-eigenvalue | −0.155136 | −0.074031 | +0.288895 | +0.156230 |

The complete per-arm full/tail NLLs, protocol, anchors, initialization hashes,
and table receipts are in the Stage A raw summary and are not merged into the
Stage B JSON block below.

### stageB_base500k_25m (base=500000.0, tokens/arm=25165824)

The runner reused the raw arm key `FMRoPE`, but at base 500,000 this arm is a
standard geometric table and its scientific identity is **Geo**, not FMRoPE.
Raw JSON keys below are retained for receipt fidelity.

| Arm | weighted OOD tail NLL | gain vs Geo |
|---|---:|---:|
| Geo (raw key `FMRoPE`) | 5.789295758463052 | 0.0 |
| anchored EVQ-Cosh | 5.71739772168509 | 0.07189803677796203 |
| phase-isotropy | 5.7671160878162775 | 0.02217967064677495 |
| min-eigenvalue | 5.676747156225618 | 0.11254860223743446 |

```json
{
  "anchored EVQ-Cosh": {
    "256": {
      "tail_minus_FMRoPE": 0.01231694221496582,
      "full_minus_FMRoPE": 0.021608829498291016
    },
    "512": {
      "tail_minus_FMRoPE": -0.12272179126739502,
      "full_minus_FMRoPE": -0.007784843444824219
    },
    "1024": {
      "tail_minus_FMRoPE": 0.07040345668792725,
      "full_minus_FMRoPE": 0.02612912654876709
    },
    "2048": {
      "tail_minus_FMRoPE": -0.15072011947631836,
      "full_minus_FMRoPE": -0.07982099056243896
    }
  },
  "phase-isotropy": {
    "256": {
      "tail_minus_FMRoPE": -0.006745457649230957,
      "full_minus_FMRoPE": 0.0026055574417114258
    },
    "512": {
      "tail_minus_FMRoPE": -0.007206320762634277,
      "full_minus_FMRoPE": 0.025563597679138184
    },
    "1024": {
      "tail_minus_FMRoPE": 0.12416684627532959,
      "full_minus_FMRoPE": 0.02595043182373047
    },
    "2048": {
      "tail_minus_FMRoPE": -0.13686323165893555,
      "full_minus_FMRoPE": -0.047450900077819824
    }
  },
  "min-eigenvalue": {
    "256": {
      "tail_minus_FMRoPE": 0.0336536169052124,
      "full_minus_FMRoPE": 0.038318753242492676
    },
    "512": {
      "tail_minus_FMRoPE": -0.06426632404327393,
      "full_minus_FMRoPE": -0.006990909576416016
    },
    "1024": {
      "tail_minus_FMRoPE": -0.022399306297302246,
      "full_minus_FMRoPE": -0.05125319957733154
    },
    "2048": {
      "tail_minus_FMRoPE": -0.20272290706634521,
      "full_minus_FMRoPE": -0.1223447322845459
    }
  }
}
```

## Runtime and matching evidence

- Device/dtype: Apple M4 Max MPS, `torch.float32`; peak driver memory was
  13,967.625 MiB for every completed arm; no OOM, NaN, CPU fallback, or other
  runtime anomaly was observed.
- Stage A arm wall-clock durations from `run_start`/`run_complete` events were
  1:05:40, 1:07:24, 1:07:37, 1:07:54, and 1:07:49 (FMRoPE, anchored
  EVQ-Cosh, phase-isotropy, pair-volume, min-eigenvalue).
- Stage B arm wall-clock durations were 0:33:41, 0:33:59, 0:34:12, and
  0:34:12 (Geo under raw key `FMRoPE`, anchored EVQ-Cosh, phase-isotropy,
  min-eigenvalue).
- Both stages used the same 50.9M model (`hidden=512`, `layers=6`, `heads=8`,
  `d_head=64`, `K=32`), seed 137, global/micro batch 256/32 with accumulation
  8, the existing AdamW schedule, deterministic local-WikiText prefix/order,
  the four frozen anchors, and the same matched-initialization SHA-256. The
  stage-specific train-prefix hashes and table hashes are preserved in the raw
  summaries and receipt.
- The determinant identity preflight passed with maximum absolute error
  `2.55351295663786e-15`; all realized frequency tables were endpoint-exact
  and strictly monotone. No result-driven protocol changes were made.

## Post-hoc 1024-anchor diagnostic

Using the already-frozen Stage B checkpoints and the same four anchors (no
training or protocol change), the phase-isotropy 1024 final-128 tail deltas
versus Geo (raw key `FMRoPE`) were `+0.174921`, `+0.222301`, `+0.068288`, and `+0.031158`
for offsets `750073`, `941884`, `950595`, and `1014394`. Thus the reversal is
not caused by one bad anchor: all four anchors move in the same direction.
The corresponding phase-isotropy full-NLL deltas were only `+0.093221`,
`+0.062483`, `−0.032957`, and `−0.018944`, averaging `+0.025950` versus the
`+0.124167` tail delta. The large aggregate reversal is therefore concentrated
near the final 128 positions of the 1024-token evaluation. This is consistent
with a length-specific phase-wrap/co-adaptation effect, but does not by itself
prove a spectral resonance mechanism.

## Construction receipt

```json
{
  "status": "PASS",
  "tables": {
    "stageB_base500k_25m": {
      "FMRoPE": {
        "base": 500000.0,
        "endpoint_exact": true,
        "strict_monotone": true,
        "float32_sha256": "23123293cf5011a65b7b217d61f42aaa57ecea912482c4ef9a6cc9bce12444cf",
        "normalized_nodes": [
          -0.0,
          0.0322580666077996,
          0.06451613013290715,
          0.09677419020657431,
          0.12903225850402925,
          0.16129032637338603,
          0.1935483852918474,
          0.22580644959784232,
          0.25806451623014004,
          0.2903225802939123,
          0.32258064088098765,
          0.35483871191698646,
          0.3870967750028413,
          0.41935483965515347,
          0.45161289983149905,
          0.483870964586007,
          0.5161290315046599,
          0.5483870975617161,
          0.5806451585310582,
          0.6129032232072678,
          0.6451612857430044,
          0.6774193523321546,
          0.7096774198406758,
          0.7419354813206469,
          0.7741935496611393,
          0.8064516145004375,
          0.8387096736928511,
          0.8709677419117533,
          0.9032258061330382,
          0.9354838688070826,
          0.9677419341189561,
          1.0000000005057124
        ]
      },
      "anchored EVQ-Cosh": {
        "base": 500000.0,
        "endpoint_exact": true,
        "strict_monotone": true,
        "float32_sha256": "41600cbe8d117570fc52673f023450b43144a2b59ddb8f4f8ca1413da4c95a21",
        "normalized_nodes": [
          -0.0,
          0.009030639513226406,
          0.018361906837860185,
          0.028014487810672565,
          0.038011275783017766,
          0.048377723614365015,
          0.05914217305739613,
          0.07033638623014651,
          0.08199606861847498,
          0.09416156041466389,
          0.10687868732405384,
          0.12019976324140755,
          0.13418493430341108,
          0.14890376910315692,
          0.16443735829108808,
          0.18088097523995986,
          0.1983476247123726,
          0.21697267716123736,
          0.23692019272903259,
          0.258391585627761,
          0.281637895388331,
          0.30697738866543783,
          0.33482179511642396,
          0.36571652562631446,
          0.4004049393631265,
          0.4399357885239943,
          0.48585345857518347,
          0.5405583420446722,
          0.6080475506116688,
          0.6955794399192053,
          0.817560873186084,
          1.0000000005057124
        ]
      },
      "phase-isotropy": {
        "base": 500000.0,
        "endpoint_exact": true,
        "strict_monotone": true,
        "float32_sha256": "045cf1a7478dd2f632f6c4da523d162d9ff59c8e159a46902e654da183cd2d56",
        "normalized_nodes": [
          -0.0,
          0.013822448132905873,
          0.027651311732848616,
          0.04148828265329404,
          0.055335324254498316,
          0.06919461613440561,
          0.08306882801573919,
          0.09696096484256843,
          0.1108747392035785,
          0.12481467854986034,
          0.1387856945659616,
          0.15279481911444207,
          0.1668489330528397,
          0.1809577286448199,
          0.19513269498442404,
          0.2093895311520567,
          0.22374192056312395,
          0.23820741051536515,
          0.25283344993503376,
          0.2676379618490377,
          0.2826519334758433,
          0.29794071891647284,
          0.31377145062209544,
          0.32975223126593495,
          0.34625370412480544,
          0.36450840211050745,
          0.38583410881660324,
          0.411880774876012,
          0.4454599116633191,
          0.4927110953445977,
          0.5728862115881918,
          1.0000000005057124
        ]
      },
      "min-eigenvalue": {
        "base": 500000.0,
        "endpoint_exact": true,
        "strict_monotone": true,
        "float32_sha256": "999d5e6eaa98df2439d0cb8dcc06fc9d680e1cc33ffeb66546157dc113b46d06",
        "normalized_nodes": [
          -0.0,
          0.014684756275798518,
          0.02937321046528051,
          0.04406640552110899,
          0.0587655220370499,
          0.0734719488561519,
          0.08818736518019613,
          0.10291368645039017,
          0.11765340700881662,
          0.13240928808708682,
          0.14718510821528213,
          0.1619849228912124,
          0.1768141171560409,
          0.19167844698380113,
          0.20658695113877565,
          0.22154952021799545,
          0.23657731245804653,
          0.25169617628918517,
          0.26691244437734046,
          0.2822574124988071,
          0.2977591327720281,
          0.31358553021642444,
          0.32950390889702497,
          0.3457294342199338,
          0.3630208132178207,
          0.38222136798673134,
          0.4043557028712783,
          0.4309807382128675,
          0.4648983539631596,
          0.5122573849691798,
          0.5922241534617793,
          1.0000000005057124
        ]
      }
    }
  },
  "identity_max_abs_error": 2.55351295663786e-15
}
```

No candidate is promoted from this matrix without a separately frozen multi-seed protocol.

## Gate verdict

**SCREEN_UNRESOLVED (internal preliminary only).** The generated raw gate string
was `FAILED_50M_GATE`, but the preregistration requires an unresolved label when
the anchored-Cosh reference is neutral or negative in the base-256 regime.
Stage A therefore cannot support a candidate-wide rejection. Stage B is a
separate base-500K/25M-token regime, not a matched support or budget ablation;
its mixed length response cannot repair that decision. These single-seed,
local-WikiText MPS results change no manuscript claim and close no method class.
