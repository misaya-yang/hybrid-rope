# Theory Appendix Extensions (L1/L2/L3 Draft Assets)

## L1) Joint Lagrangian extension
Proposed appendix objective:

`J[rho] = C[rho] - lambda * R_Burg[rho] - eta * \int_0^1 ln I(phi) dphi`

Interpretation:
- `C[rho]`: phase-collision energy (main objective)
- `R_Burg[rho]`: entropy-like regularization stabilizing density smoothness
- `I(phi)`: local resolvability/Fisher-like information proxy

Expected qualitative implication:
- positive `eta` penalizes schedules that collapse high-frequency information,
- yielding mathematically justified anchoring behavior in practical discrete settings.

## L2) Discretization error note
For inverse-CDF discretization from continuous `rho*(phi)` to `N=d/2` bins:

- define CDF `F(phi)=\int_0^phi rho(t) dt`,
- quantiles `q_j=(j+1/2)/N`,
- discrete points `phi_j=F^{-1}(q_j)`.

A practical approximation metric:
- `epsilon_q = mean_j |F(phi_j)-q_j|`

Use script:
- `scripts/import_2024/export_schedule_from_prior.py`

Outputs include `quantization_error_mean_abs_cdf`, suitable for appendix table/figure.

## L3) Checklist-compliant addenda
- Broader Impact paragraph (~150 words): include carbon footprint and long-context privacy/retrieval hallucination risk.
- Anonymous reproducibility entrypoint:
  - `scripts/import_2024/export_schedule_from_prior.py`
  - takes empirical prior and exports discrete `inv_freq` (`.json/.npy/.pt`).
