import json
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import matplotlib.pyplot as plt

CONVERSATION_DIR = Path("conversations")


def compute_statistics() -> Tuple[float, Dict[int, float]]:
    """Return overall success rate and per-step success rates."""
    total_convs = 0
    successful_convs = 0
    step_data: Dict[int, Dict[str, int]] = {}

    for file in CONVERSATION_DIR.glob("*_history.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        history = data.get("history", [])
        if not history:
            continue
        total_convs += 1

        last_res = history[-1].get("result")
        if last_res:
            last_step = last_res[-1]
            if last_step.get("is_done") and last_step.get("success"):
                successful_convs += 1

        for idx, step in enumerate(history, start=1):
            for res in step.get("result", []):
                if res.get("success") is not None:
                    info = step_data.setdefault(idx, {"success": 0, "total": 0})
                    info["total"] += 1
                    if res.get("success"):
                        info["success"] += 1

    overall_rate = (successful_convs / total_convs * 100) if total_convs else 0.0
    step_rates = {
        step: (data["success"] / data["total"] * 100 if data["total"] else 0.0)
        for step, data in step_data.items()
    }
    return overall_rate, step_rates


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Conversation Evaluation") as demo:
        gr.Markdown("# Conversation Evaluation")
        overall_text = gr.Textbox(label="Overall Success Rate", interactive=False)
        plot_output = gr.Plot(label="Success Rate per Step")
        refresh = gr.Button("Refresh")

        def refresh_metrics():
            overall, steps = compute_statistics()
            fig, ax = plt.subplots()
            if steps:
                xs = sorted(steps)
                ys = [steps[x] for x in xs]
                ax.bar(xs, ys)
                ax.set_ylim(0, 100)
                ax.set_xlabel("Step")
                ax.set_ylabel("Success Rate (%)")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            fig.tight_layout()
            return f"{overall:.2f}%", fig

        refresh.click(fn=refresh_metrics, outputs=[overall_text, plot_output])
        demo.load(refresh_metrics, outputs=[overall_text, plot_output])
    return demo
