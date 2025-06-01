# src/dea_models/visualizations.py

import matplotlib.pyplot as plt

def plot_slack_waterfall(
    slacks_in: dict[str, float],
    slacks_out: dict[str, float],
    dmu_name: str
):
    """
    Dibuja un gráfico de cascada:
      - Inputs con slack positivo (exceso) como barras negativas.
      - Outputs con slack positivo (escasez) como barras positivas.
    slacks_in: dict {input_var: slack_value}
    slacks_out: dict {output_var: slack_value}
    dmu_name: nombre de la DMU para título.
    """
    categories = list(slacks_in.keys()) + list(slacks_out.keys())
    values = [-slacks_in[k] for k in slacks_in] + [slacks_out[k] for k in slacks_out]

    fig, ax = plt.subplots(figsize=(8, 4))
    cumulative = 0
    for cat, val in zip(categories, values):
        ax.bar(cat, val, bottom=cumulative, color="C0" if val >= 0 else "C1")
        cumulative += val

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Waterfall de slacks para DMU {dmu_name}")
    ax.set_ylabel("Slack neto")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
