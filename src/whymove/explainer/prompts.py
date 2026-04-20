"""Prompt templates for the Claude explainer."""

from __future__ import annotations

from whymove.models import LabeledIntent, PositionFeatures

SYSTEM_PROMPT = """You are a chess coach explaining the strategic and tactical intent behind a specific move.

You will be given:
- The position in FEN notation
- The move played (in SAN notation)
- Detected intent labels with confidence scores
- Key position features (evaluation change, material, tactical flags)

Your explanation should:
1. Be 2-4 sentences, written for a player rated 1200-1800 ELO
2. Name and explain the primary tactical/strategic concept(s)
3. Connect the concept to concrete details from the position
4. Not repeat the move or FEN verbatim in the opening sentence
5. Use precise chess vocabulary (fork, pin, outpost, etc.)
6. Be direct and confident — state what the move achieves, not what it might achieve"""

USER_PROMPT_TEMPLATE = """\
Position (FEN): {fen}
Move played: {move_san}

Detected intents (label: confidence):
{intent_list}

Position features:
- Evaluation change: {eval_delta:+d} centipawns (positive = better for moving side)
- Material change: {material_delta:+d} centipawns
- Check: {is_check}, Capture: {is_capture}, Promotion: {is_promotion}
- Key tactical flags: {tactical_summary}

Explain the intent behind this move in 2-4 sentences.\
"""

LABELING_SYSTEM_PROMPT = """You are a chess analyst. Given a position and move, return ONLY a JSON array of applicable intent labels.
Do not include any explanation or text outside the JSON array."""

LABELING_USER_PROMPT_TEMPLATE = """\
Position (FEN): {fen}
Move: {move_san}
Eval change: {eval_delta:+d} centipawns
Is check: {is_check}, Is capture: {is_capture}
Active tactical flags: {tactical_flags}

Available labels:
{all_labels}

Return ONLY a JSON array of the most applicable labels. Example: ["fork", "mating_threat"]
Return at most 4 labels. Only include labels that clearly apply.\
"""


def format_user_prompt(
    fen: str,
    move_san: str,
    intents: list[LabeledIntent],
    features: PositionFeatures,
) -> str:
    """Format the user prompt for the explanation request."""
    intent_list = "\n".join(
        f"  - {i.label.value}: {i.confidence:.0%}" for i in intents
    ) or "  (no strong signals detected)"

    tactical_summary = _summarize_tactical_flags(features)

    return USER_PROMPT_TEMPLATE.format(
        fen=fen,
        move_san=move_san,
        intent_list=intent_list,
        eval_delta=features.eval_delta_cp,
        material_delta=features.material_delta,
        is_check=features.tactical.is_check,
        is_capture=features.tactical.is_capture,
        is_promotion=features.tactical.is_promotion,
        tactical_summary=tactical_summary or "none",
    )


def format_labeling_prompt(
    fen: str,
    move_san: str,
    features: PositionFeatures,
    all_labels: list[str],
) -> str:
    """Format the prompt for LLM-assisted label generation."""
    tactical_flags = _summarize_tactical_flags(features) or "none"
    return LABELING_USER_PROMPT_TEMPLATE.format(
        fen=fen,
        move_san=move_san,
        eval_delta=features.eval_delta_cp,
        is_check=features.tactical.is_check,
        is_capture=features.tactical.is_capture,
        tactical_flags=tactical_flags,
        all_labels=", ".join(all_labels),
    )


def _summarize_tactical_flags(features: PositionFeatures) -> str:
    """Return a comma-separated string of active tactical flags."""
    t = features.tactical
    flags = []
    if t.is_fork:
        flags.append("fork")
    if t.is_pin:
        flags.append("pin")
    if t.is_skewer:
        flags.append("skewer")
    if t.is_discovered_attack:
        flags.append("discovered_attack")
    if t.is_double_check:
        flags.append("double_check")
    if t.is_mating_threat:
        flags.append("mating_threat")
    if t.is_sacrifice:
        flags.append("sacrifice")
    if t.is_x_ray:
        flags.append("x_ray")
    if t.is_overloading:
        flags.append("overloading")
    if t.is_deflection:
        flags.append("deflection")
    if t.is_zwischenzug:
        flags.append("zwischenzug")
    if t.is_interposition:
        flags.append("interposition")
    return ", ".join(flags)
