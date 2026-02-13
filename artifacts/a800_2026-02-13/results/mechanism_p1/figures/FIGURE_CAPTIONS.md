# Figure Captions

## loss_curve_per_model.png
- Caption: 16K token-wise smoothed NLL curves for M00/M10/M01/M11.
- Legend: Model names (M00_base_orig, M10_base_hybridfreq, M01_lora_origfreq, M11_lora_hybridfreq).
- Axes: X=Token Position, Y=Smoothed NLL (window=128).

## attention_entropy_vs_position.png
- Caption: Attention entropy as a function of query token position, averaged over sampled layers and heads.
- Legend: Model names.
- Axes: X=Query Token Position, Y=Mean Attention Entropy.

## phase_collision_index_vs_length.png
- Caption: CollisionIndex(L) trend across context lengths for base_orig and hybrid.
- Legend: base_orig vs hybrid.
- Axes: X=Context Length, Y=CollisionIndex(L).

## lora_weight_diff_heatmap.png
- Caption: Layer-wise LoRA delta-weight energy distribution over low/mid/high frequency bands for Q_proj and K_proj.
- Legend: Colorbar indicates energy magnitude.
- Axes: X=Frequency Band, Y=Layer Index.
