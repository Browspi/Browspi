import os
import asyncio
import json
import logging
import re # Import re for the locator strategy enhancement
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from playwright.async_api import (
    Browser as PlaywrightBrowserInstance,
)
from playwright.async_api import (
    BrowserContext as PlaywrightContextInstance,
)
from playwright.async_api import Page
from playwright.async_api import Playwright, async_playwright
from pydantic import BaseModel, Field, model_validator, ValidationError # Added ValidationError

# Add this import
import markdownify
# -----------------------

# --- Basic Configuration Models ---

class BrowserConfig(BaseModel):
    """Basic config for the Browser."""
    headless: bool = False
    browser_args: List[str] = []

class BrowserContextConfig(BaseModel):
    """Basic config for the BrowserContext."""
    user_agent: Optional[str] = None
    viewport: Optional[Dict[str, int]] = {"width": 1280, "height": 720}
    ignore_https_errors: bool = True

# --- Simplified Browser Management ---

class Browser:
    """Manages the Playwright browser instance."""
    def __init__(self, config: BrowserConfig = BrowserConfig()):
        self.config = config
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[PlaywrightBrowserInstance] = None

    async def launch(self) -> PlaywrightBrowserInstance:
        """Launches the browser if not already running."""
        if self._browser and self._browser.is_connected(): # Check if connected
             return self._browser
        logging.info("Launching new browser instance...")
        if self._playwright is None:
             self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless, args=self.config.browser_args
            )
        except Exception as e:
             logging.error(f"Failed to launch browser: {e}")
             # Attempt to close playwright if launch fails
             if self._playwright:
                  await self._playwright.stop()
                  self._playwright = None
             raise # Re-raise the exception
        logging.info("Browser launched successfully.")
        return self._browser


    async def new_context(
        self, context_config: BrowserContextConfig = BrowserContextConfig()
    ) -> PlaywrightContextInstance:
        """Creates a new browser context."""
        browser_instance = await self.launch()
        if not browser_instance.is_connected():
             raise Exception("Browser is not connected, cannot create context.")
        context = await browser_instance.new_context(
            user_agent=context_config.user_agent,
            viewport=context_config.viewport,
            ignore_https_errors=context_config.ignore_https_errors,
        )
        return context

    async def close(self):
        """Closes the browser and stops Playwright."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logging.warning(f"Error closing browser: {e}")
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                 logging.warning(f"Error stopping playwright: {e}")
            self._playwright = None
        logging.info("Browser closed.")

# --- Simplified Agent Class ---

class AgentAction(BaseModel):
    """Represents a single action decided by the LLM."""
    action_name: str = Field(...)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = Field(None, description="Description of the target element (used for click/input).")
    element_type: Optional[str] = Field(None, description="Optional type of the element ('button', 'input', 'textarea', 'a', etc.).")

    @model_validator(mode='before')
    @classmethod
    def move_params_to_root(cls, values):
        params = values.get('parameters', {})
        if 'description' in params: values['description'] = params.pop('description')
        if 'element_type' in params: values['element_type'] = params.pop('element_type')
        return values

class LLMResponse(BaseModel):
    """Expected response structure from the LLM."""
    thought: str = Field(...)
    actions: List[AgentAction] = Field(...)

class Agent:
    """Basic agent to control the browser."""
    def __init__(self, task: str, llm: ChatOpenAI, browser: Browser):
        self.task = task
        self.llm = llm
        self.browser_manager = browser
        self._context: Optional[PlaywrightContextInstance] = None
        self._page: Optional[Page] = None
        self._history: List[Dict[str, Any]] = []

        self.system_prompt = SystemMessage(
            content=f"""
You are an AI agent controlling a web browser via Playwright to complete tasks.
Your goal is to complete the user's task: "{task}"
You will be given the current page URL, title, and a snippet of the page's content (Markdown).
Based on this state and the task, decide the next action(s).

Available Actions:
- go_to_url: Navigates to a URL. Params: {{'url': '...'}}
- click_element: Clicks an element. Provide 'description' (e.g., 'search button') and optionally 'element_type'. Params: {{'description': '...', 'element_type': 'button' (optional)}}
- input_text: Types text into an element. Provide 'description' (e.g., 'search input'), 'text', and optionally 'element_type'. Params: {{'description': '...', 'text': '...', 'element_type': 'input' (optional)}}
- scroll: Scrolls the page. Params: {{'direction': 'up'/'down', 'amount': pixels_or_'page'}}
- get_content: Extracts specific information from the current page based on a query. Params: {{'query': 'what_to_extract'}} (e.g., 'title of the first news link')
- wait: Waits. Params: {{'seconds': float}} or {{'selector': 'css_selector', 'timeout_ms': int}}
- finish: Completes the task. Params: {{'task_result': 'Final answer or summary'}}
- send_keys: Sends special keyboard keys. Params: {{'keys': 'Enter' | 'Tab' | ...}}

Respond ONLY with a valid JSON object: {{"thought": "...", "actions": [{{"action_name": "...", "parameters": {{...}}}}]}}
Describe elements clearly for 'click_element'/'input_text'. Use 'get_content' for extraction. Use 'finish' when done.
Hint: Google search bar is often a 'textarea' described as 'Search'. Use 'send_keys' with 'Enter' after input. Google News tab might be described as 'News' and could be a 'link' or 'tab' role.
"""
        )

    async def _get_context(self) -> PlaywrightContextInstance:
        if not self._context:
            self._context = await self.browser_manager.new_context()
        return self._context

    async def _get_page(self) -> Page:
        context = await self._get_context()
        if not self._page or self._page.is_closed():
             if context.pages: self._page = context.pages[-1]
             else: self._page = await context.new_page()
             # Go to blank page if new page created to ensure clean state
             if len(context.pages) == 1 and self._page.url == 'about:blank':
                 logging.info("New page created, ensuring it's ready.")
                 # Optional: Add a small wait or a dummy action if needed
                 await asyncio.sleep(0.1)
        # Ensure page is brought to front before returning
        try:
             await self._page.bring_to_front()
        except Exception as e:
             logging.warning(f"Could not bring page {self._page.url} to front: {e}. It might be closed.")
             # Attempt to get a valid page again
             if context.pages:
                  self._page = context.pages[-1]
                  await self._page.bring_to_front()
             else:
                  raise Exception("No valid pages available in context.")
        return self._page

    # --- Updated _get_page_state with content_snippet ---
    async def _get_page_state(self) -> Dict[str, Any]:
        """Gets the current state of the page, including content snippet."""
        page = await self._get_page()
        content_snippet = "Could not retrieve content." # Default
        current_url = "about:blank"
        page_title = "Untitled"
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            current_url = page.url
            page_title = await page.title()
            # Get HTML content snippet
            try:
                body_html = await page.locator('body').inner_html(timeout=5000)
                markdown_content = markdownify.markdownify(body_html)
                content_snippet = markdown_content[:2000] # Limit snippet size
            except Exception as e:
                logging.warning(f"Failed to get/process page content snippet: {e}")

        except Exception as e:
            logging.warning(f"Failed to get basic page state (URL/Title/Load): {e}")

        state = {
            "current_url": current_url,
            "page_title": page_title,
            "content_snippet": content_snippet # Send snippet to LLM
        }
        return state
    # --------------------------------------------------

    # --- Updated _execute_action Method ---
    async def _execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Executes a single action using Playwright."""
        page = await self._get_page()
        name = action.action_name
        params = action.parameters
        description = action.description
        element_type = action.element_type

        result = {"action": name, "params": {**params, "description": description, "element_type": element_type}, "status": "success", "output": ""}
        logging.info(f"Executing action: {name} with description: '{description}' (Type: {element_type}) Params: {params}")

        target_locator = None

        try:
            if name == "get_content":
                extraction_query = params.get("query", "Summarize the main content of the page.")
                logging.info(f"Attempting to extract: '{extraction_query}'")
                try:
                    # Remove erroneous timeout argument here
                    html_content = await page.content()
                    markdown_content = markdownify.markdownify(html_content)
                    page_content_for_llm = markdown_content
                    if len(page_content_for_llm) > 15000:
                        logging.warning(f"Content for extraction is long ({len(page_content_for_llm)} chars), truncating to 15000.")
                        page_content_for_llm = page_content_for_llm[:15000]
                except Exception as e:
                    logging.error(f"Failed to get or process page content for extraction: {e}")
                    result["status"] = "error"; result["output"] = f"Failed to get page content: {e}"; return result

                extraction_prompt = f"""
Based on the following page content (in Markdown format), please extract the information relevant to this query: '{extraction_query}'
Return ONLY the extracted information, without any preamble or explanation. If you cannot find the information, respond with 'Information not found.'.
Page Content:
---
{page_content_for_llm}
---
Extraction Query: '{extraction_query}'
Extracted Information:"""
                try:
                    extraction_messages = [HumanMessage(content=extraction_prompt)]
                    # Ensure self.llm is used here
                    extraction_response = await self.llm.ainvoke(extraction_messages)
                    extracted_info = str(extraction_response.content)
                    logging.info(f"LLM Extraction Result: {extracted_info}")
                    result["output"] = extracted_info if extracted_info else "LLM could not extract the requested information."
                except Exception as e:
                    logging.error(f"LLM extraction call failed: {e}"); result["status"] = "error"; result["output"] = f"LLM extraction failed: {e}"

            elif name in ["click_element", "input_text"]:
                if not description: raise ValueError("Action requires an element 'description'.")

                is_search_bar_attempt = description and "search" in description.lower()
                found_specific = False # Flag to track if specific selector worked
                if is_search_bar_attempt:
                    logging.info("Description suggests a search bar. Trying specific selectors first.")
                    specific_selectors = ['textarea[name="q"]', 'input[name="q"]','textarea[title="Search"]','input[title="Search"]','textarea[aria-label="Search"]','input[aria-label="Search"]']
                    for selector in specific_selectors:
                        try:
                            potential_locator = page.locator(selector)
                            if await potential_locator.count() > 0 and await potential_locator.is_visible(timeout=1000):
                                target_locator = potential_locator
                                logging.info(f"Found search bar using specific selector: '{selector}'")
                                found_specific = True; break
                        except Exception: continue

                if not found_specific: # If specific search failed or wasn't applicable
                     logging.info("Using general description-based locator strategies.")
                     # Regex for flexible matching (case-insensitive)
                     desc_regex = re.compile(description, re.IGNORECASE)
                     locator_strategies = [
                          lambda t=element_type, d=desc_regex: page.get_by_role(t, name=d) if t else None,
                          lambda d=description: page.get_by_placeholder(d),
                          lambda d=description: page.get_by_text(d, exact=True),
                          lambda d=description: page.get_by_label(d),
                          lambda d=description: page.get_by_title(d, exact=False),
                          # Try get_by_role without name as fallback
                          lambda t=element_type, d=description: page.get_by_role(t).filter(has_text=d) if t else None,
                          # Text based fallbacks last
                          lambda d=description: page.get_by_text(d, exact=False),
                          lambda d=description, t=element_type: page.locator(f'{t or "*"}:has-text("{d}")')
                     ]
                     for i, strategy_func in enumerate(locator_strategies):
                          try:
                                potential_locator = strategy_func()
                                if potential_locator is None: continue
                                if await potential_locator.count() > 0:
                                     if await potential_locator.first.is_visible(timeout=1500):
                                          target_locator = potential_locator.first
                                          logging.info(f"Found element using strategy #{i+1} (Visible)")
                                          break
                                     elif target_locator is None:
                                          target_locator = potential_locator.first
                                          logging.info(f"Found element using strategy #{i+1} (Visibility check timed out or hidden)")
                          except Exception as e: logging.debug(f"Locator strategy #{i+1} failed: {e}")


                if not target_locator: raise Exception(f"Could not find element described as: '{description}' (Type: {element_type})")

                await target_locator.scroll_into_view_if_needed(timeout=3000)
                try: await target_locator.wait_for(state="visible", timeout=5000)
                except Exception: logging.warning(f"Element '{description}' not strictly visible after locator found, proceeding.")

                if name == "click_element":
                    await target_locator.click(timeout=5000)
                    result["output"] = f"Clicked element described as: '{description}'"
                elif name == "input_text":
                    try: await target_locator.clear(timeout=1000)
                    except Exception: logging.warning("Could not clear element before filling.")
                    await target_locator.fill(params["text"], timeout=5000)
                    result["output"] = f"Typed '{params['text']}' into element described as: '{description}'"

            # --- Handle other actions ---
            elif name == "go_to_url": await page.goto(params["url"], wait_until="domcontentloaded", timeout=15000); result["output"] = f"Navigated to {params['url']}"
            elif name == "scroll": direction = params.get("direction", "down"); amount = params.get("amount", "page"); scroll_js = f"window.scrollBy(0, {'window.innerHeight' if amount == 'page' else (1 if direction == 'down' else -1) * int(amount)});"; await page.evaluate(scroll_js); result["output"] = f"Scrolled {direction} by {amount}"
            elif name == "wait":
                if "seconds" in params: await asyncio.sleep(params["seconds"]); result["output"] = f"Waited for {params['seconds']} seconds"
                elif "selector" in params: timeout = params.get("timeout_ms", 5000); await page.locator(params["selector"]).first.wait_for(state="visible", timeout=timeout); result["output"] = f"Waited for element {params['selector']} to be visible"
                else: await asyncio.sleep(1); result["output"] = "Waited for 1 second"
            elif name == "send_keys": await page.keyboard.press(params['keys']); result["output"] = f"Pressed key(s): {params['keys']}"
            elif name == "finish": result["output"] = params.get("task_result", "Task marked as finished."); result["is_final"] = True
            else: raise ValueError(f"Unknown action: {name}")

            if name != "wait": await asyncio.sleep(0.5)

        except Exception as e:
            logging.error(f"Error executing action {name} (Description: '{description}', Params: {params}): {e}")
            result["status"] = "error"; result["output"] = f"Failed action '{name}' on element described as '{description}': {e}"

        return result

    async def run(self, max_steps: int = 10):
        """Runs the agent loop."""
        logging.info(f"Starting agent for task: {self.task}")
        final_result = "Task could not be completed within the step limit."

        for step in range(max_steps):
            logging.info(f"--- Step {step + 1}/{max_steps} ---")
            try:
                # --- Ensure page is ready before getting state ---
                page = await self._get_page()
                try:
                    # Give page a bit more time to settle, especially after actions
                    await page.wait_for_load_state("load", timeout=5000)
                except Exception as load_err:
                     logging.warning(f"Wait for load state failed in step {step+1}: {load_err}. Getting state anyway.")
                # --------------------------------------------------

                current_state = await self._get_page_state()
                logging.info(
                    f"Current State: URL='{current_state['current_url']}', Title='{current_state['page_title']}'"
                )

                # --- Updated Prompt Content ---
                prompt_content = f"""
Current State:
URL: {current_state["current_url"]}
Title: {current_state["page_title"]}
Content Snippet (first 2000 chars of Markdown):
{current_state["content_snippet"]}

Previous History (if any):
{json.dumps(self._history[-3:], indent=2)} # Show last 3 steps

Task: {self.task}

Based on the state and task, what action(s) should be taken next? Respond ONLY with the required JSON format.
"""
                # -----------------------------
                messages = [self.system_prompt, HumanMessage(content=prompt_content)]

                response = await self.llm.ainvoke(messages)
                llm_output_str = str(response.content)

                try:
                    if llm_output_str.startswith("```json"): llm_output_str = llm_output_str[7:]
                    if llm_output_str.endswith("```"): llm_output_str = llm_output_str[:-3]
                    llm_response_data = json.loads(llm_output_str.strip())
                    if 'thought' not in llm_response_data or 'actions' not in llm_response_data: raise ValueError("LLM response missing 'thought' or 'actions' key.")
                    llm_response = LLMResponse(**llm_response_data)
                    logging.info(f"LLM Thought: {llm_response.thought}")
                except (json.JSONDecodeError, ValueError, ValidationError) as e: # Added ValidationError
                    logging.error(f"Failed to parse or validate LLM response: '{llm_output_str}'. Error: {e}")
                    self._history.append({"step": step + 1, "state": current_state, "error": f"LLM response parsing/validation failed: {e}", "llm_raw": llm_output_str})
                    final_result = "Error: Could not understand the AI's next action."; break

                step_results = []; is_finished = False
                for action in llm_response.actions:
                    action_result = await self._execute_action(action)
                    step_results.append(action_result)
                    if action_result["status"] == "error":
                        logging.warning(f"Action '{action.action_name}' failed. Stopping sequence for this step."); break
                    if action.action_name == "finish":
                        final_result = action_result["output"]; is_finished = True; break

                self._history.append({"step": step + 1, "state": current_state, "thought": llm_response.thought, "results": step_results})
                if is_finished: logging.info("Task finished."); break

            except Exception as e:
                logging.exception(f"Critical error during step {step + 1}: {e}"); self._history.append({"step": step + 1, "error": str(e)}); final_result = f"Error during step {step + 1}: {e}"; break
        else: logging.warning("Max steps reached.")

        print("\n--- Agent Run Summary ---"); print(f"Task: {self.task}"); print(f"Final Result/Status: {final_result}")
        return final_result

# --- Main Execution ---
async def main_async(task: str):
    load_dotenv(); api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise ValueError("OPENAI_API_KEY not found")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0.0) # Changed temp back to 0
    browser_manager = Browser(config=BrowserConfig(headless=False))
    agent = Agent(task=task, llm=llm, browser=browser_manager)
    final_output = "Agent run did not complete as expected."
    try: final_output = await agent.run(max_steps=15)
    except Exception as e: logging.exception("Agent run failed critically."); final_output = f"Agent run failed with error: {e}"
    finally: input("Press Enter to close browser..."); await browser_manager.close()
    print(f"\nFINAL OUTPUT:\n{final_output}")

if __name__ == "__main__":
    default_task = "Go to google.com, search for 'latest AI news', and return the title of the first news link found."
    import sys
    custom_task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(main_async(task=custom_task or default_task))