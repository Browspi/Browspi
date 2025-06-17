# src/browspi/ui/app.py

import gradio as gr

from browspi.ui.manager import UIManager

ui_manager = UIManager()


def create_interface():
    """Tạo và trả về Gradio interface."""
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="blue"), title="Browspi Agent"
    ) as interface:
        gr.Markdown("# Browspi Web Agent")

        with gr.Row():
            # --- Cột cho các Input ---
            with gr.Column(scale=2):
                task_input = gr.Textbox(
                    label="Your Task",
                    placeholder="Example: Find the latest news about AI...",
                )
                session_name_input = gr.Textbox(
                    label="Session Name (Optional)",
                    placeholder="e.g., 'research-ai-news'. Leave blank for random.",
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
                    # --- THAY ĐỔI TẠI ĐÂY ---
                    # Xóa max_actions và slow_mo, thay bằng max_steps
                    max_steps_slider = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1, label="Max Steps"
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

        # Cập nhật danh sách inputs theo đúng thứ tự mới
        inputs = [
            task_input,
            session_name_input,
            llm_provider_dropdown,
            browser_profile_dropdown,
            use_vision_checkbox,
            max_steps_slider,  # <-- Input mới
        ]

        start_button.click(
            fn=ui_manager.start_automation_task,
            inputs=inputs,
            outputs=[final_summary_output, history_log_output],
        )

    return interface


def main():
    """Hàm chính để khởi chạy ứng dụng web."""
    app = create_interface()
    app.launch(server_name="localhost", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
