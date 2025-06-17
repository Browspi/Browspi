# src/browspi/ui/manager.py

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI

from browspi.main import ActionManager, AgentHistoryList, AutomationConfig, WebAutomator
from browspi.services.browser.service import DEFAULT_BROWSER_PROFILE
from browspi.ui.browser_utils import (
    get_chrome_executable_path,
    get_persistent_profile_path,
)


class UIManager:
    def start_automation_task(
        self,
        task: str,
        session_name_from_ui: str,
        llm_provider: str,
        browser_profile_name: str,
        use_vision: bool,
        max_steps: int,
    ) -> tuple[str, str]:
        if not task:
            return "Task is empty.", ""

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not found.", ""
        os.environ["OPENAI_API_KEY"] = api_key

        if not session_name_from_ui:
            slug_task = re.sub(r"[^a-zA-Z0-9_-]", "", task.replace(" ", "-"))[:50]
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
                current_bp = DEFAULT_BROWSER_PROFILE.model_copy(
                    update={
                        "user_data_dir": persistent_profile_path,
                        "executable_path": chrome_exe_path,
                        "headless": False,
                        "args": (DEFAULT_BROWSER_PROFILE.args or [])
                        + ["--start-maximized", "--disable-gpu", "--no-sandbox"],
                    }
                )
            else:
                current_bp = DEFAULT_BROWSER_PROFILE

            conversation_dir = Path("conversations")
            conversation_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(conversation_dir / f"{final_session_name}_history.json")

            current_as = AutomationConfig(
                use_vision=use_vision,
                max_actions_per_step=2,
                tool_calling_method="tools",
                page_extraction_llm=llm,
                save_conversation_path=save_path,
                max_failures=2,
            )

            agent = WebAutomator(
                task=task,
                llm=llm,
                agent_settings=current_as,
                browser_profile=current_bp,
                controller=ActionManager(),
            )

            history: AgentHistoryList = await agent.start_task(max_steps=max_steps)

            # --- BẮT ĐẦU LOGIC FORMAT OUTPUT GIỐNG CONSOLE ---
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
                            if res_item.extracted_content:
                                log_lines.append(
                                    f"  > Action Result: Extracted content length {len(res_item.extracted_content)} chars."
                                )
                            if res_item.error:
                                log_lines.append(f"  > Action Error: {res_item.error}")

                    browser_s = hist_item.state or {}
                    if browser_s.get("url"):
                        log_lines.append(f"  (Browser at: {browser_s.get('url')})")
                    log_lines.append("")  # Thêm một dòng trống để phân cách

            # --- KẾT THÚC LOGIC FORMAT OUTPUT ---

            final_content = history.final_result()
            if final_content:
                final_summary = f"✅ Final Result: {final_content}"
            elif history.is_done() and not history.is_successful():
                final_summary = "❌ Task marked as done, but reported failure."
            else:
                final_summary = f"❓ Task finished after {len(history.history)} steps without a final result."

            return final_summary, "\n".join(log_lines)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            final_summary, history_log = loop.run_until_complete(run_async_flow())
            return final_summary, history_log
        except Exception as e:
            print(f"An error occurred during agent execution: {type(e).__name__} - {e}")
            return f"Error: {type(e).__name__} - {e}", ""
