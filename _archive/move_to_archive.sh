#!/bin/bash
# 将非主线文件移入 _archive/
# ⚠️ 请先 review 本脚本再执行！不会删除任何文件，只是移动。

set -e
REPO="/sessions/eloquent-optimistic-hopper/mnt/hybrid-rope"
ARCHIVE="$REPO/_archive"

echo "=== 移动顶层废弃目录 ==="
for dir in sigmoid_rope_experiments tmp_phase4_compare batch_report_2026-02-23_downstream_eval batch_report_2026-02-24_new_server_scan neurips_plan experiments archives; do
    if [ -d "$REPO/$dir" ]; then
        echo "  Moving $dir → _archive/"
        mv "$REPO/$dir" "$ARCHIVE/"
    fi
done

echo "=== 移动 results/ 中的非主线数据 ==="
mkdir -p "$ARCHIVE/results_legacy"
for dir in advisor_package_2026-02-15 anchored_sigmoid_v3_followup archive_low_priority attention_distribution baseline_passkey comprehensive_theta cross_model_wikitext_v1 eval_700m frequency_range_analysis gamma_search hybrid_comparison hybrid_comparison_v2 llama13b_triangle_boundary llama8b_fair_lora_suite_20260214 llama8b_post_eval_20260214 llama_shape_theta_min llama_theta_matched_shape_control niah_llama3_base_full night_run_anchored_x20_9h optimal_base_search our_method_comparison passkey_long phase4_passkey_sanity phase4_passkey_sanity_50m_rerun1 phase_collision_comparison phase_collision_comparison_v2 phase_transition qwen_3way_compare qwen_comparison qwen_hybrid_lora qwen_int4_vs_base_only qwen_plugandplay_wikitext_v1 rope_scaling_v2 theoretical_validation theory_validation theory_2026-02-22 train_freq_comparison unified_search unified_search_3cfg_3seed _weights_quarantine evidence_chain_50m_3cfg3seed; do
    if [ -d "$REPO/results/$dir" ]; then
        echo "  Moving results/$dir → _archive/results_legacy/"
        mv "$REPO/results/$dir" "$ARCHIVE/results_legacy/"
    fi
done

echo "=== 移动 results/paper_ready 中已归入 submission 的目录 ==="
# paper_ready 下的非 evq_tau_sweep 数据移入归档
mkdir -p "$ARCHIVE/results_paper_ready_legacy"
for dir in baseline_passkey comprehensive_theta cross_model_wikitext_v1 eval_700m hybrid_comparison hybrid_comparison_v2 llama13b_triangle_boundary llama8b_fair_lora_suite_20260214 llama8b_post_eval_20260214 llama_shape_theta_min llama_theta_matched_shape_control niah_llama3_base_full night_run_anchored_x20_9h our_method_comparison passkey_long qwen_3way_compare qwen_comparison qwen_hybrid_lora qwen_int4_vs_base_only qwen_plugandplay_wikitext_v1 rope_scaling_v2 train_freq_comparison unified_search unified_search_3cfg_3seed; do
    if [ -d "$REPO/results/paper_ready/$dir" ]; then
        echo "  Moving results/paper_ready/$dir → _archive/results_paper_ready_legacy/"
        mv "$REPO/results/paper_ready/$dir" "$ARCHIVE/results_paper_ready_legacy/"
    fi
done

echo "=== 移动废弃顶层文件 ==="
for file in import.md AI_HANDOFF.md; do
    if [ -f "$REPO/$file" ]; then
        echo "  Moving $file → _archive/"
        mv "$REPO/$file" "$ARCHIVE/"
    fi
done

echo ""
echo "✅ 归档完成！"
echo "主线代码和数据未受影响。"
echo "如需恢复，从 _archive/ 中移回即可。"
