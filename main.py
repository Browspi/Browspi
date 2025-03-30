import os
import asyncio
import json
import logging
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
from pydantic import BaseModel, Field

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
        if self._browser:
            return self._browser
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless, args=self.config.browser_args
        )
        return self._browser

    async def new_context(
        self, context_config: BrowserContextConfig = BrowserContextConfig()
    ) -> PlaywrightContextInstance:
        """Creates a new browser context."""
        browser_instance = await self.launch()
        context = await browser_instance.new_context(
            user_agent=context_config.user_agent,
            viewport=context_config.viewport,
            ignore_https_errors=context_config.ignore_https_errors,
        )
        return context

    async def close(self):
        """Closes the browser and stops Playwright."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logging.info("Browser closed.")


# --- Simplified Agent Class ---


class AgentAction(BaseModel):
    """Represents a single action decided by the LLM."""

    action_name: str = Field(
        ...,
        description="The name of the action to perform (e.g., 'go_to_url', 'click_element', 'input_text', 'get_content', 'finish')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action (e.g., {'url': '...'}, {'selector': '...', 'text': '...'}, {'task_result': '...'})",
    )


class LLMResponse(BaseModel):
    """Expected response structure from the LLM."""

    thought: str = Field(..., description="Your reasoning for the chosen action(s).")
    actions: List[AgentAction] = Field(
        ..., description="A list of actions to execute in sequence."
    )


class Agent:
    """Basic agent to control the browser."""

    def __init__(self, task: str, llm: ChatOpenAI, browser: Browser):
        self.task = task
        self.llm = llm
        self.browser_manager = browser
        self._context: Optional[PlaywrightContextInstance] = None
        self._page: Optional[Page] = None
        self._history: List[Dict[str, Any]] = []  # Keep track of steps

        # Simplified system prompt
        self.system_prompt = SystemMessage(
            content=f"""
You are an AI agent controlling a web browser via Playwright to complete tasks.
Your goal is to complete the user's task: "{task}"

You will be given the current page URL, title, and sometimes page content or interactive elements.
Based on this state and the task, decide the next action(s).

Available Actions:
- go_to_url: Navigates to a URL. Params: {{'url': '...'}}
- click_element: Clicks an element. Params: {{'selector': 'css_or_xpath_selector'}}
- input_text: Types text into an element. Params: {{'selector': 'css_or_xpath_selector', 'text': '...'}}
- scroll: Scrolls the page. Params: {{'direction': 'up'/'down', 'amount': pixels_or_'page'}}
- get_content: Extracts content from the current page. Params: {{'query': 'what_to_extract'}} (Optional: If no query, summarize)
- wait: Waits for a specified time or element. Params: {{'seconds': float}} or {{'selector': 'css_selector', 'timeout_ms': int}}
- finish: Completes the task. Params: {{'task_result': 'Final answer or summary'}}

Respond ONLY with a valid JSON object in the following format:
{{
  "thought": "Your reasoning for the next action(s).",
  "actions": [
    {{"action_name": "action1", "parameters": {{...}}}},
    {{"action_name": "action2", "parameters": {{...}}}}
  ]
}}
Be precise with selectors. If unsure, use get_content first. Indicate task completion with the 'finish' action.
"""
        )

    async def _get_context(self) -> PlaywrightContextInstance:
        """Initializes or returns the browser context."""
        if not self._context:
            self._context = await self.browser_manager.new_context()
        return self._context

    async def _get_page(self) -> Page:
        """Gets the current page, creating one if necessary."""
        context = await self._get_context()
        if not self._page or self._page.is_closed():
            if context.pages:
                self._page = context.pages[-1]  # Use the most recent page
            else:
                self._page = await context.new_page()
        await self._page.bring_to_front()
        return self._page

    async def _get_page_state(self) -> Dict[str, Any]:
        """Gets the current state of the page."""
        page = await self._get_page()
        await page.wait_for_load_state("domcontentloaded", timeout=5000)  # Short wait
        # Basic state: URL and Title
        state = {
            "current_url": page.url,
            "page_title": await page.title(),
            # "content_snippet": await page.content()[:1000] # Optional: add a snippet
        }
        return state

    async def _execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Executes a single action using Playwright."""
        page = await self._get_page()
        name = action.action_name
        params = action.parameters
        result = {"action": name, "params": params, "status": "success", "output": ""}
        logging.info(f"Executing action: {name} with params: {params}")

        try:
            if name == "go_to_url":
                await page.goto(
                    params["url"], wait_until="domcontentloaded", timeout=15000
                )
                result["output"] = f"Navigated to {params['url']}"
            elif name == "click_element":
                await page.locator(params["selector"]).click(timeout=5000)
                result["output"] = f"Clicked element: {params['selector']}"
            elif name == "input_text":
                await page.locator(params["selector"]).fill(
                    params["text"], timeout=5000
                )
                result["output"] = (
                    f"Typed '{params['text']}' into: {params['selector']}"
                )
            elif name == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", "page")
                if amount == "page":
                    scroll_js = f"window.scrollBy(0, {'window.innerHeight' if direction == 'down' else '-window.innerHeight'});"
                else:
                    scroll_js = f"window.scrollBy(0, {(1 if direction == 'down' else -1) * amount});"
                await page.evaluate(scroll_js)
                result["output"] = f"Scrolled {direction} by {amount}"
            elif name == "get_content":
                # Basic content extraction (could be improved with LLM summary)
                content = await page.content()
                result["output"] = content[:2000]  # Limit content size
                result["full_content_available"] = True  # Indicate more content exists
            elif name == "wait":
                if "seconds" in params:
                    await asyncio.sleep(params["seconds"])
                    result["output"] = f"Waited for {params['seconds']} seconds"
                elif "selector" in params:
                    timeout = params.get("timeout_ms", 5000)
                    await page.locator(params["selector"]).wait_for(
                        state="visible", timeout=timeout
                    )
                    result["output"] = (
                        f"Waited for element {params['selector']} to be visible"
                    )
                else:
                    await asyncio.sleep(1)  # Default wait
                    result["output"] = "Waited for 1 second"
            elif name == "finish":
                result["output"] = params.get("task_result", "Task marked as finished.")
                result["is_final"] = True
            else:
                raise ValueError(f"Unknown action: {name}")

            # Short delay after action to allow UI updates
            await asyncio.sleep(0.5)

        except Exception as e:
            logging.error(f"Error executing action {name}: {e}")
            result["status"] = "error"
            result["output"] = str(e)

        return result

    async def run(self, max_steps: int = 10):
        """Runs the agent loop."""
        logging.info(f"Starting agent for task: {self.task}")
        final_result = "Task could not be completed within the step limit."

        for step in range(max_steps):
            logging.info(f"--- Step {step + 1}/{max_steps} ---")
            try:
                current_state = await self._get_page_state()
                logging.info(
                    f"Current State: URL='{current_state['current_url']}', Title='{current_state['page_title']}'"
                )

                prompt_content = f"""
Current State:
URL: {current_state["current_url"]}
Title: {current_state["page_title"]}

Previous History (if any):
{json.dumps(self._history[-3:], indent=2)} # Show last 3 steps

Task: {self.task}

Based on the state and task, what action(s) should be taken next? Respond ONLY with the required JSON format.
"""
                messages = [self.system_prompt, HumanMessage(content=prompt_content)]

                # Call LLM
                response = await self.llm.ainvoke(messages)
                llm_output_str = response.content

                # Parse LLM Response
                try:
                    # Clean potential markdown code fences
                    if llm_output_str.startswith("```json"):
                        llm_output_str = llm_output_str[7:]
                    if llm_output_str.endswith("```"):
                        llm_output_str = llm_output_str[:-3]

                    llm_response_data = json.loads(llm_output_str.strip())
                    llm_response = LLMResponse(**llm_response_data)
                    logging.info(f"LLM Thought: {llm_response.thought}")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse LLM response: {e}")
                    llm_response = LLMResponse(
                        thought="Error parsing response", actions=[]
                    )
                    llm_response.actions = [
                        AgentAction(
                            action_name="finish",
                            parameters={"task_result": "Error parsing response"},
                        )
                    ]

                step_results = []
                is_finished = False
                for action in llm_response.actions:
                    action_result = await self._execute_action(action)
                    step_results.append(action_result)
                    if action_result["status"] == "error":
                        logging.warning(
                            f"Action '{action.action_name}' failed. Stopping sequence for this step."
                        )
                        break  # Stop executing further actions in this step if one fails
                    if action.action_name == "finish":
                        final_result = action_result["output"]
                        is_finished = True
                        break  # Stop if finish action is called

                self._history.append(
                    {
                        "step": step + 1,
                        "state": current_state,
                        "thought": llm_response.thought,
                        "results": step_results,
                    }
                )

                if is_finished:
                    logging.info("Task finished.")
                    break

            except Exception as e:
                logging.exception(f"Critical error during step {step + 1}: {e}")
                self._history.append({"step": step + 1, "error": str(e)})
                final_result = f"Error during step {step + 1}: {e}"
                break
        else:
            logging.warning("Max steps reached.")

        # Print final result and history
        print("\n--- Agent Run Summary ---")
        print(f"Task: {self.task}")
        print(f"Final Result/Status: {final_result}")
        # print("\n--- History ---")
        # print(json.dumps(self._history, indent=2))
        return final_result


# --- Main Execution ---


async def main_async(task: str):
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize components
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0)
    browser_manager = Browser(
        config=BrowserConfig(headless=False)
    )  # Run headful for debugging
    agent = Agent(task=task, llm=llm, browser=browser_manager)

    final_output = "Agent run did not complete as expected."
    try:
        final_output = await agent.run(max_steps=15)
    except Exception as e:
        logging.exception("Agent run failed critically.")
        final_output = f"Agent run failed with error: {e}"
    finally:
        input("Press Enter to close browser...")  # Keep browser open until user input
        await browser_manager.close()

    print(f"\nFINAL OUTPUT:\n{final_output}")


if __name__ == "__main__":
    default_task = "Go to google.com, search for 'latest AI news', and return the title of the first news link found."
    # Example: Get task from command line or use default
    import sys

    custom_task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(main_async(task=custom_task or default_task))
