# src/browspi/ui/manager.py

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI

from browspi.main import (
    ActionManager,
    AgentHistoryList,
    AutomationConfig,
    WebAutomator,
)
from browspi.services.browser.service import DEFAULT_BROWSER_PROFILE
from browspi.ui.browser_utils import (
    get_chrome_executable_path,
    get_persistent_profile_path,
)

# Import hàm cấu hình từ file use case mới
from browspi.ui.use_cases.linkedin_apply import get_linkedin_task_config
# ADDED: Import the news controller
from browspi.init_agent.news import new_controller


class UIManager:
    def start_automation_task(
        self,
        task_type: str,
        task_prompt: str, # This will serve as 'topic' for News Research
        session_name_from_ui: str,
        llm_provider: str,
        browser_profile_name: str,
        use_vision: bool,
        max_steps: int,
    ) -> tuple[str, str]:
        # --- THAY ĐỔI: Logic được đơn giản hóa rất nhiều ---
        if task_type == "LinkedIn Job Application":
            final_task, controller, error = get_linkedin_task_config()
            if error:
                return error, ""

            # Bắt buộc sử dụng trình duyệt có thể thấy để xử lý đăng nhập/captcha
            browser_profile_name = "Persistent (Visible)"
            print(
                "INFO: LinkedIn task selected. Forcing 'Persistent (Visible)' browser profile."
            )
        # ADDED: New elif condition for News Research
        elif task_type == "News Research":
            if not task_prompt: # task_prompt here is the 'topic'
                return "Topic for news research is empty.", ""
            
            # Construct the task prompt for the News agent
            final_task = f"""
            Research the latest developments regarding {task_prompt} from at least 5 different reputable news sources.

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
            controller = new_controller # Use the specific controller from news.py
            print(f"INFO: News Research task selected for topic: {task_prompt}.")
        else: # General Task
            if not task_prompt:
                return "Task is empty.", ""
            final_task = task_prompt
            controller = ActionManager() # Sử dụng controller mặc định

        # --- Phần còn lại của file giữ nguyên ---

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not found.", ""

        if not session_name_from_ui:
            slug_task = re.sub(r"[^a-zA-Z0-9_-]", "", final_task.replace(" ", "-"))[:50]
            final_session_name = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug_task}"
            )
        else:
            final_session_name = session_name_from_ui

        async def run_async_flow():
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

            if browser_profile_name == "Persistent (Visible)":
                persistent_profile_path = get_persistent_profile_path()
                chrome_exe_path = get_chrome_executable_path()
                if not persistent_profile_path or not chrome_exe_path:
                    raise FileNotFoundError(
                        "For 'Persistent (Visible)' profile, please set PERSISTENT_PROFILE_PATH and CHROME_EXE_PATH in your .env file."
                    )
                current_bp = DEFAULT_BROWSER_PROFILE.model_copy(
                    update={
                        "user_data_dir": persistent_profile_path,
                        "executable_path": chrome_exe_path,
                        "headless": False,
                        "args": (DEFAULT_BROWSER_PROFILE.args or [])
                        + ["--start-maximized"],
                    }
                )
            else:
                current_bp = DEFAULT_BROWSER_PROFILE

            conversation_dir = Path("conversations")
            conversation_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(conversation_dir / f"{final_session_name}_history.json")

            current_as = AutomationConfig(
                use_vision=use_vision,
                max_actions_per_step=3,
                tool_calling_method="tools",
                page_extraction_llm=llm,
                save_conversation_path=save_path,
                max_failures=3,
            )

            agent = WebAutomator(
                task=final_task,
                llm=llm,
                agent_settings=current_as,
                browser_profile=current_bp,
                controller=controller,  # Sử dụng controller tương ứng
            )

            history: AgentHistoryList = await agent.start_task(max_steps=max_steps)
            await agent.close()

            # --- Logic format output ---
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
                                log_lines.append(
                                    f"  > Action Result: {summary.strip()}"
                                )

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

            return final_summary, "\n".join(log_lines)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            final_summary, history_log = loop.run_until_complete(run_async_flow())
            return final_summary, history_log
        except Exception as e:
            print(f"An error occurred during agent execution: {type(e).__name__} - {e}")
            return f"Error: {type(e).__name__} - {e}", ""