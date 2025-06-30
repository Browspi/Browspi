# src/browspi/ui/app.py

import gradio as gr
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from browspi.ui.manager import UIManager

ui_manager = UIManager()


def get_conversation_files():
    """Reads all conversation history JSON files from the 'conversations' directory."""
    conversation_dir = Path("conversations")
    if not conversation_dir.exists():
        return []
    return list(conversation_dir.glob("*_history.json"))


def calculate_success_rates():
    """
    Calculates the overall success rate and per-step success rates for each conversation.
    """
    files = get_conversation_files()
    if not files:
        return 0.0, {}

    overall_success_count = 0
    per_conversation_step_success = {}

    for file in files:
        data = None
        # Try to open with different encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file, "r", encoding=encoding) as f:
                    data = json.load(f)
                break  # If successful, no need to try other encodings
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
        if data:
            try:
                history = data.get("history", [])

                # Overall success
                is_done = history and history[-1].get("result") and history[-1]["result"][-1].get("is_done")
                if is_done and history[-1]["result"][-1].get("success"):
                    overall_success_count += 1

                # Per-step success
                step_successes = []
                for step in history:
                    results = step.get("result", [])
                    step_successful = all(not r.get("error") for r in results)
                    step_successes.append(step_successful)
                # Use a cleaner name for the conversation key and only add if there are steps
                if step_successes:
                    convo_name = file.stem.replace('_history', '')
                    per_conversation_step_success[convo_name] = step_successes

            except Exception as e:
                print(f"Error processing file {file.name}: {e}")
                continue


    overall_success_rate = (overall_success_count / len(files)) * 100 if files else 0.0
    return overall_success_rate, per_conversation_step_success


def render_evaluation_plots():
    """
    Renders the plots for overall success rate and per-step success rate.
    """
    overall_success_rate, per_step_success = calculate_success_rates()

    # Overall Success Rate Plot (Pie Chart)
    fig_overall, ax_overall = plt.subplots()
    labels = 'Success', 'Failure'
    sizes = [overall_success_rate, 100 - overall_success_rate]
    colors = ['green', 'red']
    if sum(sizes) > 0:
        ax_overall.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    else:
        ax_overall.pie([1], labels=["No data"], colors=["grey"])
    ax_overall.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax_overall.set_title('Overall Conversation Success Rate')


    # Per-Step Success Rate Pie Charts for each conversation
    if per_step_success:
        num_convos = len(per_step_success)
        # Determine grid size for subplots
        cols = 2
        rows = math.ceil(num_convos / cols)
        fig_steps, axes = plt.subplots(rows, cols, figsize=(10, rows * 5), squeeze=False)
        axes = axes.flatten()  # Flatten to make it easier to iterate

        i = 0
        for convo_name, successes in per_step_success.items():
            ax = axes[i]
            num_steps = len(successes)
            
            if num_steps > 0:
                success_count = sum(successes)
                failure_count = num_steps - success_count
                
                step_labels = 'Successful Steps', 'Failed Steps'
                step_sizes = [success_count, failure_count]
                step_colors = ['#90ee90', '#ffcccb']  # Light green, light red

                if sum(step_sizes) > 0:
                    ax.pie(step_sizes, labels=step_labels, colors=step_colors, autopct='%1.1f%%', startangle=90)
                else:
                    ax.pie([1], labels=["No steps"], colors=["grey"])
                
                ax.set_title(f"{convo_name}\n({success_count}/{num_steps} successful steps)")
            else:
                ax.text(0.5, 0.5, "No steps in conversation.", ha='center', va='center')
                ax.set_title(convo_name)

            ax.axis('equal')
            i += 1

        # Hide any unused subplots
        for j in range(i, len(axes)):
            axes[j].set_visible(False)

        fig_steps.suptitle('Per-Step Success Rate in Each Conversation', fontsize=16)
        fig_steps.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    else:
        fig_steps, ax_steps = plt.subplots()
        ax_steps.text(0.5, 0.5, "No conversation data to display.", ha='center', va='center')
        ax_steps.set_title('Per-Step Success Rate')


    return fig_overall, fig_steps


def create_interface():
    """Tạo và trả về Gradio interface."""
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="blue"), title="Browspi Agent"
    ) as interface:
        gr.Markdown("# Browspi Web Agent")

        with gr.Tabs():
            with gr.TabItem("Agent"):
                with gr.Row():
                    # --- Cột cho các Input ---
                    with gr.Column(scale=2):
                        # --- THAY ĐỔI: Thêm lựa chọn loại tác vụ ---
                        task_type_radio = gr.Radio(
                            label="Task Type",
                            choices=["General Task", "LinkedIn Job Application", "News Research"], # ADDED "News Research"
                            value="General Task",
                        )

                        task_input = gr.Textbox(
                            label="Your Task", # Default label
                            placeholder="Example: Find the latest news about AI...", # Default placeholder
                            visible=True,
                        )
                        session_name_input = gr.Textbox(
                            label="Session Name (Optional)",
                            placeholder="e.g., 'linkedin-apply'. Leave blank for random.",
                            info="A history file with this name will be saved.",
                        )

                        with gr.Accordion("LLM Settings", open=True):
                            llm_provider_dropdown = gr.Dropdown(
                                label="LLM Provider",
                                choices=["OpenAI", "Mistral"],
                                value="OpenAI",
                            )

                        with gr.Accordion("Browser Settings", open=True):
                            browser_profile_dropdown = gr.Dropdown(
                                label="Browser Profile",
                                choices=["Persistent (Visible)", "Default (Headless)"],
                                value="Persistent (Visible)",
                            )

                        with gr.Accordion("Core & Advanced Settings", open=True):
                            use_vision_checkbox = gr.Checkbox(
                                label="Use Vision (Analyze screenshots)", value=True
                            )
                            max_steps_slider = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1, label="Max Steps"
                            )

                        start_button = gr.Button("Start Task", variant="primary")

                    # --- Cột cho các Output ---
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        final_summary_output = gr.Textbox(
                            label="Final Summary", interactive=False, lines=4
                        )
                        history_log_output = gr.Textbox(
                            label="Execution History", interactive=False, lines=15
                        )

                # --- THAY ĐỔI: Thêm logic để ẩn/hiện và đổi label task_input ---
                def toggle_task_input(task_type):
                    if task_type == "LinkedIn Job Application":
                        return gr.update(visible=False, value="", label="Your Task")
                    elif task_type == "News Research":
                        return gr.update(visible=True, label="Research Topic", placeholder="Example: Covid-19 in Vietnam") # Update label and placeholder
                    else: # General Task
                        return gr.update(visible=True, label="Your Task", placeholder="Example: Find the latest news about AI...")

                task_type_radio.change(
                    fn=toggle_task_input,
                    inputs=task_type_radio,
                    outputs=task_input,
                )

                # Cập nhật danh sách inputs theo đúng thứ tự mới
                inputs = [
                    task_type_radio,
                    task_input,
                    session_name_input,
                    llm_provider_dropdown,
                    browser_profile_dropdown,
                    use_vision_checkbox,
                    max_steps_slider,
                ]

                start_button.click(
                    fn=ui_manager.start_automation_task,
                    inputs=inputs,
                    outputs=[final_summary_output, history_log_output],
                )
            with gr.TabItem("Evaluate"):
                with gr.Column():
                    gr.Markdown("## Conversation Evaluation")
                    evaluate_button = gr.Button("Generate Evaluation Plots")
                    with gr.Row():
                        overall_plot = gr.Plot(label="Overall Success Rate")
                    with gr.Row():
                        steps_plot = gr.Plot(label="Per-Step Success Rate")
                    
                    evaluate_button.click(
                        fn=render_evaluation_plots,
                        outputs=[overall_plot, steps_plot]
                    )


    return interface


def main():
    """Hàm chính để khởi chạy ứng dụng web."""
    app = create_interface()
    app.launch(server_name="localhost", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()