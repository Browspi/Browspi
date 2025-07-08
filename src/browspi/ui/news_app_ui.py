import gradio as gr
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path

# Import necessary components from browspi
from langchain_openai import ChatOpenAI
from browspi.main import AutomationConfig, WebAutomator, DEFAULT_BROWSER_PROFILE, logging
from browspi.ui.browser_utils import get_chrome_executable_path, get_persistent_profile_path

# Import the specific controller and models from the news agent
# We import new_controller directly because it contains the registered actions like save_news_data
from browspi.init_agent.news import new_controller, New, save_news_data

logger = logging.getLogger(__name__)


# Wrapper function to run the news task
async def run_news_task(
    topic: str,
    max_steps: int,
    session_name_from_ui: str,
    llm_provider: str,
    browser_profile_name: str,
    use_vision: bool,
) -> tuple[str, str]:
    """
    Runs the news research automation task based on the provided UI inputs.
    """
    # Check for OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OPENAI_API_KEY environment variable not found. Please set it in your .env file.", ""

    # Initialize LLM based on selected provider
    llm = None
    if llm_provider == "OpenAI":
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        except Exception as e:
            return f"Error initializing OpenAI LLM: {e}", ""
    elif llm_provider == "Mistral":
        # Mistral LLM integration would go here if supported by the project structure
        # For now, it's a placeholder as per llm_utils.py
        return "Mistral LLM not yet implemented for this UI. Please choose OpenAI.", ""
    else:
        return f"Unsupported LLM provider: {llm_provider}", ""

    # Construct the task prompt dynamically using the provided topic
    task = f"""
    Research the latest developments regarding {topic} from at least 3 different reputable news sources.

    For each source:
    1. Navigate to their search function and find articles about the topic from the past week
    2. Extract the headline, publication date, and author, and save the link to the article with save_news_data function.
    3. Summarize the key points in 2-3 sentences

    After gathering information, synthesize the findings into a comprehensive summary that notes any differences in reporting or perspective between the sources.

    Tips:
    - If you encounter a captcha please wait for 10 seconds before retrying.
    - Try to search on google first to find the latest articles.
    - Don't treat a google preview as a full article, always click through to the original source.
    
    Ensure that the summary is concise and highlights the most significant developments.
    """

    # Configure browser settings based on UI selection
    current_bp = DEFAULT_BROWSER_PROFILE.model_copy()
    if browser_profile_name == "Persistent (Visible)":
        persistent_profile_path = get_persistent_profile_path()
        chrome_exe_path = get_chrome_executable_path()
        if not persistent_profile_path or not chrome_exe_path:
            return "For 'Persistent (Visible)' profile, please set PERSISTENT_PROFILE_PATH and CHROME_EXE_PATH in your .env file.", ""
        current_bp.user_data_dir = persistent_profile_path
        current_bp.executable_path = chrome_exe_path
        current_bp.headless = False
        current_bp.args = (current_bp.args or []) + ["--start-maximized"]
    else:  # Default (Headless)
        current_bp.headless = True
        current_bp.user_data_dir = None
        current_bp.executable_path = None

    # Handle session name for saving conversation history
    if not session_name_from_ui:
        # Create a URL-friendly slug from the topic
        slug_topic = "".join(c if c.isalnum() or c in ['-', '_'] else '-' for c in topic.replace(" ", "-"))[:50]
        final_session_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug_topic}"
    else:
        final_session_name = session_name_from_ui

    conversation_dir = Path("conversations")
    conversation_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(conversation_dir / f"{final_session_name}_history.json")

    # Configure automation settings
    current_as = AutomationConfig(
        use_vision=use_vision,
        max_actions_per_step=3,
        tool_calling_method="tools",
        page_extraction_llm=llm,
        save_conversation_path=save_path,
        max_failures=3,
    )

    # Initialize WebAutomator with the imported new_controller from news.py
    agent = WebAutomator(
        task=task,
        llm=llm,
        agent_settings=current_as,
        browser_profile=current_bp,
        controller=new_controller,  # Use the pre-configured controller from news.py
    )

    final_summary = ""
    history_log = ""
    try:
        # Start the automation task
        history = await agent.start_task(max_steps=max_steps)
        await agent.close()

        # Format execution history for display in the UI
        log_lines = []
        if history.history:
            for i, hist_item in enumerate(history.history):
                step_metadata = hist_item.metadata or {}
                log_lines.append(f"--- Step {step_metadata.get('step', i + 1)} ---")

                if hist_item.model_output:
                    cs = hist_item.model_output.current_state
                    log_lines.append(f"👍 LLM Eval: {cs.evaluation_previous_goal}")
                    log_lines.append(f"🧠 LLM Memory: {cs.memory}")
                    log_lines.append(f"🎯 LLM Next Goal: {cs.next_goal}")

                    for action_idx, action_item in enumerate(
                        hist_item.model_output.action
                    ):
                        ad = action_item.model_dump(
                            exclude_unset=True, exclude_none=True
                        )
                        if ad:
                            action_name = list(ad.keys())[0]
                            action_params = json.dumps(list(ad.values())[0])
                            log_lines.append(
                                f"🛠️ LLM Action {action_idx + 1}/{len(hist_item.model_output.action)}: {action_name}({action_params})"
                            )

                if hist_item.result:
                    for res_idx, res_item in enumerate(hist_item.result):
                        summary = ""
                        if res_item.extracted_content:
                            summary += (
                                f"Content: '{res_item.extracted_content[:100]}...'"
                            )
                        if res_item.error:
                            summary += f" Error: {res_item.error}"
                        if summary:
                            log_lines.append(f"  > Action Result: {summary.strip()}")

                browser_s = hist_item.state or {}
                if browser_s.get("url"):
                    log_lines.append(f"  (Browser at: {browser_s.get('url')})")
                log_lines.append("")

        final_content = history.final_result()
        if final_content:
            final_summary = f"✅ Final Result: {final_content}"
        elif history.is_done() and not history.is_successful():
            final_summary = "❌ Task marked as done, but reported failure."
        else:
            final_summary = f"❓ Task finished after {len(history.history)} steps without a final result."

        history_log = "\n".join(log_lines)

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {type(e).__name__} - {e}")
        final_summary = f"Error: {type(e).__name__} - {e}"
        history_log = (
            ""  # Clear history log on error if it's incomplete or misleading
        )

    return final_summary, history_log


def create_interface():
    """
    Creates and returns the Gradio interface for the News Research Agent.
    """
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="green"), title="Browspi News Agent"
    ) as interface:
        gr.Markdown("# Browspi Web Agent: News Research")

        with gr.Row():
            # Column for input controls
            with gr.Column(scale=2):
                gr.Markdown("### Configure Your News Research Task")
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="Example: Covid-19 in Vietnam",
                    value="Covid-19 in Vietnam",  # Default value as per news.py
                    info="The topic for news research.",
                )
                session_name_input = gr.Textbox(
                    label="Session Name (Optional)",
                    placeholder="e.g., 'covid-news-research'. Leave blank for random.",
                    info="A history file with this name will be saved in the 'conversations' directory.",
                )

                with gr.Accordion("LLM Settings", open=True):
                    llm_provider_dropdown = gr.Dropdown(
                        label="LLM Provider",
                        choices=["OpenAI", "Mistral"],
                        value="OpenAI",
                        interactive=True,
                        info="Select the Large Language Model provider to use.",
                    )

                with gr.Accordion("Browser Settings", open=True):
                    browser_profile_dropdown = gr.Dropdown(
                        label="Browser Profile",
                        choices=["Persistent (Visible)", "Default (Headless)"],
                        value="Persistent (Visible)",
                        interactive=True,
                        info="Persistent (Visible) requires CHROME_EXE_PATH and PERSISTENT_PROFILE_PATH in .env.",
                    )

                with gr.Accordion("Core & Advanced Settings", open=True):
                    use_vision_checkbox = gr.Checkbox(
                        label="Use Vision (Analyze screenshots)",
                        value=True,
                        interactive=True,
                        info="Enable visual analysis of web pages for better understanding.",
                    )
                    max_steps_slider = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=50,  # Default value as per news.py
                        step=1,
                        label="Max Steps",
                        interactive=True,
                        info="Maximum number of steps the agent will attempt to complete the task.",
                    )

                start_button = gr.Button("Start News Research", variant="primary")

            # Column for output displays
            with gr.Column(scale=3):
                gr.Markdown("### Agent Execution Results")
                final_summary_output = gr.Textbox(
                    label="Final Summary",
                    interactive=False,
                    lines=4,
                    show_copy_button=True,
                )
                history_log_output = gr.Textbox(
                    label="Execution History",
                    interactive=False,
                    lines=15,
                    show_copy_button=True,
                    autoscroll=True,
                )

        # Define inputs for the click event
        inputs = [
            topic_input,
            max_steps_slider,
            session_name_input,
            llm_provider_dropdown,
            browser_profile_dropdown,
            use_vision_checkbox,
        ]

        # Link the button click to the asynchronous task execution
        start_button.click(
            fn=run_news_task,
            inputs=inputs,
            outputs=[final_summary_output, history_log_output],
        )

    return interface


def main():
    """
    Main function to launch the Gradio web application.
    """
    app = create_interface()
    app.launch(server_name="localhost", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()