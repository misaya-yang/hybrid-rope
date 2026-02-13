#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/Users/misaya.yanghejazfs.com.au/dfrope/github_bundle/artifacts/a800_2026-02-13/results')
MECH_P1 = ROOT / 'mechanism_p1'
H800_FOLLOWUP = ROOT / 'h800_3h_followup' / 'results.json'
H800_POLY = ROOT / 'h800_3h_poly_followup' / 'results.json'
R6000_ROOT = Path('/Users/misaya.yanghejazfs.com.au/dfrope/github_bundle/artifacts/r6000_2026-02-13')
LENGTH_KEYS = ['2048', '4096', '8192', '12288', '14336', '16384']

OUT = ROOT / 'mechanism_p2_framework'
FIG = OUT / 'figures'


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def avg_attention_metrics(attn: Dict[str, Any], model_name: str) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for _layer, heads in attn[model_name].items():
        for _head, m in heads.items():
            rows.append(m)
    def mkey(k: str) -> float:
        return float(np.mean([float(r[k]) for r in rows]))
    return {
        'entropy': mkey('entropy'),
        'sink_mass': mkey('sink_mass'),
        'long_range_mass': mkey('long_range_mass'),
        'mean_distance': mkey('mean_distance'),
    }


def extract_frequency_runs(h800_main: Dict[str, Any], h800_poly: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for name, rec in h800_main['variants'].items():
        ev = rec['eval']
        out[name] = {
            'group': rec.get('group', 'unknown'),
            'source': 'h800_3h_followup',
            'ppl_by_length': {
                L: {
                    'sequential': float(ev[L]['sequential']['mean']),
                    'random_start_mean': float(ev[L]['random_start']['mean_over_seeds']),
                    'random_start_std_seed': float(ev[L]['random_start']['std_over_seeds']),
                }
                for L in LENGTH_KEYS
            },
            'collapse_ratio_16k_over_2k': float(
                ev['16384']['random_start']['mean_over_seeds'] / max(ev['2048']['random_start']['mean_over_seeds'], 1e-12)
            ),
        }

    for name, rec in h800_poly['variants'].items():
        if 'error' in rec:
            continue
        ev = rec['eval']
        out[name] = {
            'group': rec.get('group', 'poly'),
            'source': 'h800_3h_poly_followup',
            'ppl_by_length': {
                L: {
                    'sequential': float(ev[L]['sequential']['mean']),
                    'random_start_mean': float(ev[L]['random_start']['mean_over_seeds']),
                    'random_start_std_seed': float(ev[L]['random_start']['std_over_seeds']),
                }
                for L in LENGTH_KEYS
            },
            'collapse_ratio_16k_over_2k': float(
                ev['16384']['random_start']['mean_over_seeds'] / max(ev['2048']['random_start']['mean_over_seeds'], 1e-12)
            ),
        }

    return out


def build_results() -> Dict[str, Any]:
    factor = load_json(MECH_P1 / '2x2_factor_results.json')
    loss = load_json(MECH_P1 / 'loss_curve_per_model.json')
    attn = load_json(MECH_P1 / 'attention_stats.json')
    phase = load_json(MECH_P1 / 'phase_collision_index.json')
    lora = load_json(MECH_P1 / 'lora_weight_diff.json')

    h800 = load_json(H800_FOLLOWUP)
    poly = load_json(H800_POLY)

    freq = extract_frequency_runs(h800, poly)

    a_base = avg_attention_metrics(attn, 'M00_base_orig')
    a_hybrid = avg_attention_metrics(attn, 'M11_lora_hybridfreq')

    cross_arch = {
        'qwen_eval_suite_json_found': bool((R6000_ROOT / 'results' / 'qwen_hybrid_lora' / 'eval_suite.json').exists()),
        'qwen_status_file_found': bool((R6000_ROOT / 'R6000_QWEN_HYBRID_STATUS_2026-02-13.md').exists()),
        'status': 'pending_qwen_eval_output_sync',
    }

    return {
        'meta': {
            'name': 'mechanism_p2_framework',
            'description': 'Unified stability/mechanism framework output from A800 runs',
            'lengths': [int(v) for v in LENGTH_KEYS],
            'slicings': ['sequential', 'random_start'],
            'seeds': [42, 123, 777],
        },
        'experiments': {
            '1_frequency_shape_vs_stability': {
                'variants': freq,
            },
            '2_factor_2x2_ablation': factor,
            '3_token_wise_loss_curve': {
                'base': loss['M00_base_orig'],
                'hybrid': loss['M11_lora_hybridfreq'],
            },
            '4_attention_stats': {
                'base': a_base,
                'hybrid': a_hybrid,
                'raw': {
                    'M00_base_orig': attn['M00_base_orig'],
                    'M11_lora_hybridfreq': attn['M11_lora_hybridfreq'],
                },
            },
            '5_phase_collision_index': phase,
            '6_lora_weight_freq_diff': lora,
            '7_cross_arch_validation': cross_arch,
        },
    }


def plot_frequency_16k(freq: Dict[str, Any]) -> None:
    items = sorted(
        [(n, v['ppl_by_length']['16384']['random_start_mean']) for n, v in freq.items()],
        key=lambda x: x[1],
    )
    names = [x[0] for x in items]
    vals = [x[1] for x in items]

    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=25, ha='right')
    plt.ylabel('PPL@16K (random_start mean over seeds)')
    plt.xlabel('Variant')
    plt.title('Frequency Shape vs 16K Stability')
    plt.tight_layout()
    plt.savefig(FIG / 'frequency_shape_16k_bar.png', dpi=180)
    plt.close()


def plot_frequency_curves(freq: Dict[str, Any]) -> None:
    x = [int(v) for v in LENGTH_KEYS]

    plt.figure(figsize=(10, 5))
    for name, rec in sorted(freq.items()):
        y = [rec['ppl_by_length'][L]['random_start_mean'] for L in LENGTH_KEYS]
        plt.plot(x, y, marker='o', label=name)

    plt.xlabel('Context Length')
    plt.ylabel('PPL (random_start mean over seeds)')
    plt.title('Multi-Length Stability Curves by Frequency Shape')
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / 'frequency_shape_multi_length.png', dpi=180)
    plt.close()


def plot_factor_2x2_16k(factor: Dict[str, Any]) -> None:
    names = ['M00_base_orig', 'M10_base_hybridfreq', 'M01_lora_origfreq', 'M11_lora_hybridfreq']
    seq = [factor[n]['16384']['sequential']['ppl'] for n in names]
    rnd = [factor[n]['16384']['random_start']['ppl'] for n in names]

    x = np.arange(len(names))
    w = 0.36

    plt.figure(figsize=(8.5, 4.5))
    plt.bar(x - w / 2, seq, width=w, label='sequential')
    plt.bar(x + w / 2, rnd, width=w, label='random_start')
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylabel('PPL@16K')
    plt.xlabel('2x2 Model Variant')
    plt.title('2x2 Factor Ablation at 16K')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'factor_2x2_16k.png', dpi=180)
    plt.close()


def plot_tokenwise_loss(loss: Dict[str, Any]) -> None:
    b = loss['M00_base_orig']
    h = loss['M11_lora_hybridfreq']
    x = b['positions']

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, b['smoothed_nll'], label='base (M00)')
    plt.plot(x, h['smoothed_nll'], label='hybrid+LoRA (M11)')
    plt.xlabel('Token Position')
    plt.ylabel('Smoothed NLL')
    plt.title('Token-wise Loss Curve at 16K')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'tokenwise_loss_base_vs_hybrid.png', dpi=180)
    plt.close()


def plot_attention(attn_base: Dict[str, float], attn_hybrid: Dict[str, float]) -> None:
    keys = ['entropy', 'sink_mass', 'long_range_mass', 'mean_distance']
    b = [attn_base[k] for k in keys]
    h = [attn_hybrid[k] for k in keys]

    x = np.arange(len(keys))
    w = 0.36

    plt.figure(figsize=(8.5, 4.5))
    plt.bar(x - w / 2, b, width=w, label='base')
    plt.bar(x + w / 2, h, width=w, label='hybrid')
    plt.xticks(x, keys, rotation=15)
    plt.ylabel('Metric Value')
    plt.xlabel('Attention Metric')
    plt.title('Attention Collapse Metrics at 16K (avg over sampled layers/heads)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'attention_metrics_base_vs_hybrid.png', dpi=180)
    plt.close()


def plot_phase(phase: Dict[str, Any]) -> None:
    lengths = sorted([int(k) for k in phase.keys()])
    base = [phase[str(L)]['base_orig'] for L in lengths]
    hybrid = [phase[str(L)]['hybrid'] for L in lengths]

    plt.figure(figsize=(8.5, 4.5))
    plt.plot(lengths, base, marker='o', label='base_orig')
    plt.plot(lengths, hybrid, marker='o', label='hybrid')
    plt.xlabel('Context Length')
    plt.ylabel('CollisionIndex(L)')
    plt.title('Phase Collision Index vs Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'phase_collision_vs_length.png', dpi=180)
    plt.close()


def plot_lora_heatmap(lora: Dict[str, Any]) -> None:
    layers = sorted([int(k) for k in lora.keys()])
    q = np.array([[lora[str(i)]['Q_proj']['low_freq_energy'], lora[str(i)]['Q_proj']['mid_freq_energy'], lora[str(i)]['Q_proj']['high_freq_energy']] for i in layers])
    k = np.array([[lora[str(i)]['K_proj']['low_freq_energy'], lora[str(i)]['K_proj']['mid_freq_energy'], lora[str(i)]['K_proj']['high_freq_energy']] for i in layers])

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5), constrained_layout=True)
    im0 = axs[0].imshow(q, aspect='auto')
    axs[0].set_title('Q_proj energy')
    axs[0].set_xlabel('Frequency Band')
    axs[0].set_ylabel('Layer')
    axs[0].set_xticks([0, 1, 2], ['low', 'mid', 'high'])
    axs[0].set_yticks(range(len(layers)), [str(i) for i in layers])
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(k, aspect='auto')
    axs[1].set_title('K_proj energy')
    axs[1].set_xlabel('Frequency Band')
    axs[1].set_ylabel('Layer')
    axs[1].set_xticks([0, 1, 2], ['low', 'mid', 'high'])
    axs[1].set_yticks(range(len(layers)), [str(i) for i in layers])
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    fig.suptitle('LoRA Weight Energy Distribution by Frequency Band')
    plt.savefig(FIG / 'lora_weight_freq_heatmap.png', dpi=180)
    plt.close()


def build_summary(results: Dict[str, Any]) -> str:
    freq = results['experiments']['1_frequency_shape_vs_stability']['variants']
    rows = sorted(
        [(n, r['ppl_by_length']['16384']['random_start_mean'], r['collapse_ratio_16k_over_2k']) for n, r in freq.items()],
        key=lambda x: x[1],
    )

    factor = results['experiments']['2_factor_2x2_ablation']
    m00_16 = factor['M00_base_orig']['16384']['sequential']['ppl']
    m11_16 = factor['M11_lora_hybridfreq']['16384']['sequential']['ppl']

    attn = results['experiments']['4_attention_stats']
    phase = results['experiments']['5_phase_collision_index']
    lora = results['experiments']['6_lora_weight_freq_diff']

    q_low = np.mean([lora[k]['Q_proj']['low_freq_energy'] for k in lora.keys()])
    q_mid = np.mean([lora[k]['Q_proj']['mid_freq_energy'] for k in lora.keys()])
    q_high = np.mean([lora[k]['Q_proj']['high_freq_energy'] for k in lora.keys()])

    k_low = np.mean([lora[k]['K_proj']['low_freq_energy'] for k in lora.keys()])
    k_mid = np.mean([lora[k]['K_proj']['mid_freq_energy'] for k in lora.keys()])
    k_high = np.mean([lora[k]['K_proj']['high_freq_energy'] for k in lora.keys()])

    lines: List[str] = []
    lines.append('Mechanism P2 Framework Summary')
    lines.append('')
    lines.append('## 1) 频谱形状与稳定性（A800, TinyStories from-scratch）')
    lines.append('')
    lines.append('| rank | variant | ppl@16k(rand) | collapse_ratio(16k/2k) |')
    lines.append('|---:|---|---:|---:|')
    for i, (n, p16, cr) in enumerate(rows, 1):
        lines.append(f'| {i} | {n} | {p16:.3f} | {cr:.3f} |')

    lines.append('')
    lines.append('结论：`sigmoid_th100k` 最优，`poly` 明显优于 `geo_10k_baseline`，但弱于最优 sigmoid/high-theta 方案。')

    lines.append('')
    lines.append('## 2) 2x2 因子消融（LLaMA-3-8B）')
    lines.append('')
    lines.append(f'- M00@16K(seq): {m00_16:.3f}')
    lines.append(f'- M11@16K(seq): {m11_16:.3f}')
    lines.append(f'- 提升倍数 (M00/M11): {m00_16/max(m11_16,1e-9):.2f}x')
    lines.append('- 结论：频谱+LoRA 耦合带来最大稳定化收益。')

    lines.append('')
    lines.append('## 3) Token-wise Loss / Attention / Collision / LoRA 证据')
    lines.append('')
    lines.append(f"- attention entropy: base={attn['base']['entropy']:.4f}, hybrid={attn['hybrid']['entropy']:.4f}")
    lines.append(f"- sink_mass: base={attn['base']['sink_mass']:.4f}, hybrid={attn['hybrid']['sink_mass']:.4f}")
    lines.append(f"- phase collision @16K: base={phase['16384']['base_orig']:.6f}, hybrid={phase['16384']['hybrid']:.6f}")
    lines.append(f'- LoRA Q energy(low/mid/high): {q_low:.4f}/{q_mid:.4f}/{q_high:.4f}')
    lines.append(f'- LoRA K energy(low/mid/high): {k_low:.4f}/{k_mid:.4f}/{k_high:.4f}')
    lines.append('- 结论：指标整体一致支持“结构性外推失稳 + 频谱形状可调控稳定边界”。')

    lines.append('')
    lines.append('## 4) 跨架构状态')
    lines.append('')
    ca = results['experiments']['7_cross_arch_validation']
    lines.append(f"- qwen_eval_suite_json_found: {ca['qwen_eval_suite_json_found']}")
    lines.append(f"- status: {ca['status']}")
    lines.append('- 说明：当前仓库内仅有 Qwen 运行状态与脚本，缺少最终 eval_suite.json 同步。')

    lines.append('')
    lines.append('## 5) 下一步执行建议')
    lines.append('')
    lines.append('1. 同协议补齐 Qwen eval_suite.json，并复用本框架脚本生成跨架构对比图。')
    lines.append('2. 对 top-2 频谱（sigmoid_th100k, sigmoid_th500k）做训练 seed 重训复现（>=3 seeds）。')
    lines.append('3. 在 LLaMA 7B 上做 poly/sigmoid 的统一频谱 sweep，闭环验证 tiny->7B 一致性。')

    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    results = build_results()

    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    (OUT / 'summary.md').write_text(build_summary(results))

    freq = results['experiments']['1_frequency_shape_vs_stability']['variants']
    factor = results['experiments']['2_factor_2x2_ablation']
    loss = load_json(MECH_P1 / 'loss_curve_per_model.json')
    attn = results['experiments']['4_attention_stats']
    phase = results['experiments']['5_phase_collision_index']
    lora = results['experiments']['6_lora_weight_freq_diff']

    plot_frequency_16k(freq)
    plot_frequency_curves(freq)
    plot_factor_2x2_16k(factor)
    plot_tokenwise_loss(loss)
    plot_attention(attn['base'], attn['hybrid'])
    plot_phase(phase)
    plot_lora_heatmap(lora)

    (OUT / 'figures' / 'README.md').write_text(
        '\n'.join(
            [
                '# mechanism_p2_framework figures',
                '',
                '- `frequency_shape_16k_bar.png`',
                '- `frequency_shape_multi_length.png`',
                '- `factor_2x2_16k.png`',
                '- `tokenwise_loss_base_vs_hybrid.png`',
                '- `attention_metrics_base_vs_hybrid.png`',
                '- `phase_collision_vs_length.png`',
                '- `lora_weight_freq_heatmap.png`',
            ]
        )
        + '\n'
    )

    print('[done] wrote', OUT / 'results.json')
    print('[done] wrote', OUT / 'summary.md')


if __name__ == '__main__':
    main()
