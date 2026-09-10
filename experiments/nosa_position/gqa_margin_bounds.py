"""Conditional GQA ranking bounds; callers must supply valid positive Z bounds.

This module does not obtain those bounds and is not a deployed selector.
"""
import torch


def shared_denominator_margin_lower(mass_a, mass_b, z_lower, z_upper):
    if not bool(((z_lower > 0) & (z_upper >= z_lower)).all()):
        raise ValueError('denominator bounds must be positive and ordered')
    difference=mass_a-mass_b
    return torch.where(difference >= 0,difference/z_upper,difference/z_lower).sum(-1)


def independent_score_margin_lower(mass_a,mass_b,z_lower,z_upper):
    return (mass_a/z_upper-mass_b/z_lower).sum(-1)
