import os
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)  # Added for type hinting LLM
from langchain_openai import ChatOpenAI
from playwright.async_api import (
    Browser as PlaywrightBrowser,
    BrowserContext as PlaywrightContext,
    Page as PlaywrightPage,
    Locator,
    Playwright,
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)
from pydantic import BaseModel, Field, model_validator, ValidationError
import markdownify

# --- Configuration Models ---


class BrowserProfileConfig(BaseModel):
    headless: bool = False
    executable_path: Optional[str] = None
    persistent_user_data_dir: Optional[str] = None
    browser_args: List[str] = Field(default_factory=list)
    user_agent: Optional[str] = None
    viewport: Optional[Dict[str, int]] = {"width": 1280, "height": 720}
    ignore_https_errors: bool = True
    default_navigation_timeout: float = 30000  # ms
    default_timeout: float = 10000  # ms

    def playwright_launch_options(self) -> Dict[str, Any]:
        opts = {
            "headless": self.headless,
            "args": list(self.browser_args or []),
            "executable_path": self.executable_path,
        }
        return {k: v for k, v in opts.items() if v is not None}

    def playwright_context_options(self) -> Dict[str, Any]:
        opts = {
            "user_agent": self.user_agent,
            "viewport": self.viewport,
            "ignore_https_errors": self.ignore_https_errors,
        }
        return {k: v for k, v in opts.items() if v is not None}


# --- Browser Session Management ---


class BrowserSession:
    def __init__(self, profile_config: BrowserProfileConfig = BrowserProfileConfig()):
        self.profile_config = profile_config
        self._playwright_instance: Optional[Playwright] = None
        self._browser: Optional[PlaywrightBrowser] = None
        self._context: Optional[PlaywrightContext] = None
        self._current_page: Optional[PlaywrightPage] = None
        self._is_persistent_context_active: bool = False

    async def _start_playwright(self) -> Playwright:
        if self._playwright_instance is None:
            self._playwright_instance = await async_playwright().start()
        return self._playwright_instance

    async def get_context(self) -> PlaywrightContext:
        playwright = await self._start_playwright()
        if self._context:
            try:
                _ = self._context.pages
                return self._context
            except PlaywrightError:
                logging.warning(
                    "Existing context is no longer usable. Creating a new one."
                )
                await self._close_context_internals()

        if self.profile_config.persistent_user_data_dir:
            logging.info(
                f"Launching persistent context: {self.profile_config.persistent_user_data_dir}"
            )
            launch_opts = self.profile_config.playwright_launch_options()
            context_opts = self.profile_config.playwright_context_options()

            # Args for launch_persistent_context itself
            persistent_context_specific_args = {
                "headless": launch_opts.pop("headless", self.profile_config.headless),
                "executable_path": launch_opts.pop(
                    "executable_path", self.profile_config.executable_path
                ),
            }

            # Combine remaining launch_opts (like browser_args) with context_opts
            combined_options = {
                **launch_opts,
                **context_opts,
                **persistent_context_specific_args,
            }
            # Filter out --user-data-dir from args as it's handled by launch_persistent_context's first param
            if "args" in combined_options:
                combined_options["args"] = [
                    arg
                    for arg in combined_options["args"]
                    if not arg.startswith("--user-data-dir=")
                    and not arg.startswith("--profile-directory=")
                ]

            final_launch_options = {
                k: v for k, v in combined_options.items() if v is not None
            }

            try:
                self._context = await playwright.chromium.launch_persistent_context(
                    self.profile_config.persistent_user_data_dir, **final_launch_options
                )
                self._browser = self._context.browser
                self._is_persistent_context_active = True
            except Exception as e:
                logging.error(
                    f"Failed to launch persistent context: {type(e).__name__} - {e}"
                )
                raise
        else:
            if not self._browser or not self._browser.is_connected():
                launch_opts = self.profile_config.playwright_launch_options()
                logging.info(
                    f"Launching new browser instance with options: {launch_opts}"
                )
                self._browser = await playwright.chromium.launch(**launch_opts)

            context_opts = self.profile_config.playwright_context_options()
            logging.info(
                f"Creating new regular browser context with options: {context_opts}"
            )
            self._context = await self._browser.new_context(**context_opts)
            self._is_persistent_context_active = False

        return self._context

    async def get_current_page(self) -> PlaywrightPage:
        context = await self.get_context()
        page_is_invalid = True
        if self._current_page and self._current_page.context == context:
            try:
                if not self._current_page.is_closed():
                    page_is_invalid = False
            except PlaywrightError:
                pass

        if page_is_invalid:
            if context.pages:
                self._current_page = context.pages[-1]
            else:
                self._current_page = await context.new_page()

        if self._current_page.is_closed():  # Re-check and create if closed
            self._current_page = await context.new_page()

        if (
            not self._current_page.is_closed()
            and self._current_page.url == "about:blank"
        ):
            await self._current_page.wait_for_timeout(100)

        try:
            if not self._current_page.is_closed():
                await self._current_page.bring_to_front()
        except PlaywrightError as e:
            logging.warning(f"Could not bring page to front: {e}. Re-creating page.")
            self._current_page = await context.new_page()  # Fallback: create new page
            if (
                not self._current_page.is_closed()
                and self._current_page.url == "about:blank"
            ):
                await self._current_page.wait_for_timeout(100)
            await self._current_page.bring_to_front()  # Try again

        return self._current_page

    async def _close_context_internals(self):
        if self._context:
            try:
                await self._context.close()
            except PlaywrightError:
                pass
        self._context = None
        self._current_page = None

        if (
            self._browser
            and not self._is_persistent_context_active
            and self._browser.is_connected()
        ):
            try:
                await self._browser.close()
            except PlaywrightError:
                pass
        self._browser = (
            None  # Reset browser if it was tied to the closed non-persistent context
        )

    async def close(self):
        logging.info("Closing BrowserSession resources.")
        await (
            self._close_context_internals()
        )  # Closes context and potentially non-persistent browser

        # If browser was from persistent context, closing context should have handled it.
        # If _browser still exists and is connected, it might be an issue or a shared browser not to be closed here.
        # For this scoped BrowserSession, we assume its browser is either from persistent or self-launched.
        if self._browser and self._browser.is_connected():
            logging.warning(
                "Browser instance still connected after context closure attempts; trying to close explicitly."
            )
            try:
                await self._browser.close()
            except PlaywrightError:
                pass
        self._browser = None

        if self._playwright_instance:
            try:
                await self._playwright_instance.stop()
            except Exception:
                pass
            self._playwright_instance = None
        logging.info("BrowserSession resources closed.")

    async def __aenter__(self):
        await self._start_playwright()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# --- Action Definitions ---
class GoToUrlParams(BaseModel):
    url: str


class ClickElementParams(BaseModel):
    description: str
    element_type: Optional[str] = None


class InputTextParams(BaseModel):
    description: str
    text: str
    element_type: Optional[str] = None


class ScrollParams(BaseModel):
    direction: str = Field(default="down", pattern="^(up|down)$")
    amount: Union[str, int] = "page"


class GetContentParams(BaseModel):
    query: str


class WaitParams(BaseModel):
    seconds: Optional[float] = None
    selector: Optional[str] = None
    timeout_ms: Optional[int] = Field(default=5000, alias="timeout")


class FinishParams(BaseModel):
    task_result: str


class SendKeysParams(BaseModel):
    keys: str


class ActionResult(BaseModel):
    action_name: str
    params: Dict[str, Any]
    status: str = "success"
    output: Optional[str] = None
    error: Optional[str] = None
    is_final: bool = False


class ActionDefinition(BaseModel):
    name: str
    description: str
    param_model: Optional[type[BaseModel]] = None
    func: Callable[..., Awaitable[ActionResult]]


# --- Controller ---
class Controller:
    def __init__(self):
        self.actions: Dict[str, ActionDefinition] = {}

    def register_action(
        self, name: str, description: str, param_model: Optional[type[BaseModel]] = None
    ):
        def decorator(func: Callable[..., Awaitable[ActionResult]]):
            self.actions[name] = ActionDefinition(
                name=name, description=description, param_model=param_model, func=func
            )
            return func

        return decorator

    def get_prompt_description(self) -> str:
        descriptions = []
        for name, action_def in self.actions.items():
            param_info_parts = []
            if action_def.param_model:
                try:
                    schema = action_def.param_model.model_json_schema()
                    props = schema.get("properties", {})
                    required_props = schema.get("required", [])
                    for p_name, p_info in props.items():
                        p_type = p_info.get("type", "any")
                        p_desc = p_info.get("description", "")
                        is_req = " (required)" if p_name in required_props else ""
                        param_info_parts.append(
                            f"{p_name}: {p_type} ('{p_desc}'){is_req}"
                        )
                except Exception:
                    pass  # Fallback

            param_info_str = ""
            if param_info_parts:
                # Escape curly braces for .format() by doubling them
                param_info_str = f"Params: {{{{{', '.join(param_info_parts)}}}}}"
            else:
                param_info_str = "Params: None"

            descriptions.append(f"- {name}: {action_def.description}. {param_info_str}")
        return "\n".join(descriptions)

    async def execute(
        self, action_name: str, params_dict: Dict[str, Any], session: BrowserSession
    ) -> ActionResult:
        action_def = self.actions.get(action_name)
        if not action_def:
            return ActionResult(
                action_name=action_name,
                params=params_dict,
                status="error",
                error=f"Unknown action: {action_name}",
            )

        validated_params_model = None
        if action_def.param_model:
            try:
                validated_params_model = action_def.param_model(**params_dict)
            except ValidationError as e:
                return ActionResult(
                    action_name=action_name,
                    params=params_dict,
                    status="error",
                    error=f"Invalid parameters for {action_name}: {e}",
                )

        try:
            if validated_params_model:
                return await action_def.func(
                    session=session, params=validated_params_model
                )
            else:  # Handles actions with no params or if param_model is None
                return await action_def.func(
                    session=session, **params_dict
                )  # Splat if func takes them directly
        except PlaywrightError as pwe:
            error_msg = f"Playwright error executing {action_name}: {type(pwe).__name__} - {pwe}"
            logging.error(error_msg)
            return ActionResult(
                action_name=action_name,
                params=params_dict,
                status="error",
                error=error_msg,
            )
        except Exception as e:
            error_msg = (
                f"Error executing action {action_name}: {type(e).__name__} - {e}"
            )
            logging.error(error_msg, exc_info=True)
            return ActionResult(
                action_name=action_name,
                params=params_dict,
                status="error",
                error=error_msg,
            )


# --- Agent Pydantic Models & State ---
class AgentLLMAction(BaseModel):
    action_name: str = Field(...)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentLLMResponse(BaseModel):
    thought: str
    actions: List[AgentLLMAction]


class PageState(BaseModel):
    url: str
    title: str
    content_snippet: str  # Markdown


class AgentHistoryEntry(BaseModel):
    step: int
    page_state: Optional[PageState] = None
    llm_response: Optional[AgentLLMResponse] = None
    action_results: List[ActionResult] = Field(default_factory=list)
    step_error: Optional[str] = None


# --- Agent ---
class Agent:
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: BrowserSession,
        controller: Controller,
    ):
        self.task = task
        self.llm = llm
        self.browser_session = browser_session
        self.controller = controller
        self.history: List[AgentHistoryEntry] = []

        # IMPORTANT: This is a regular string. {task} and {available_actions} are for .format().
        # Literal braces for the JSON example are DOUBLED: {{ and }}
        self.system_prompt_template = """You are an AI agent controlling a web browser. Your goal is to complete the user's task: "{task}"
You will be given the current page URL, title, and a snippet of its content (Markdown).
Based on this, decide the next action(s).

Available Actions:
{available_actions}

Respond ONLY with a valid JSON object using the following schema: {{"thought": "your reasoning for the actions", "actions": [{{"action_name": "action_to_take", "parameters": {{"param_name": "param_value"}}}}]}}
- The "actions" list can contain one or more actions.
- Describe elements clearly for actions like 'click_element' or 'input_text'.
- Use 'get_content' for information extraction.
- Use 'finish' with the final answer when the entire task is verifiably complete.
"""

    async def _get_page_state_summary(self) -> PageState:
        page = await self.browser_session.get_current_page()
        content_snippet, current_url, page_title = (
            "Error retrieving content.",
            "about:error",
            "Error Page",
        )
        try:
            if page.is_closed():
                raise PlaywrightError("Page is closed.")
            await page.wait_for_load_state(
                "domcontentloaded",
                timeout=self.browser_session.profile_config.default_navigation_timeout
                / 3,
            )
            current_url = page.url
            page_title = await page.title()
            body_html = await page.locator("body").inner_html(
                timeout=self.browser_session.profile_config.default_timeout
            )
            content_snippet = markdownify.markdownify(body_html)[
                :4000
            ]  # Increased snippet
        except PlaywrightError as e:
            logging.warning(
                f"PlaywrightError getting page state: {type(e).__name__} - {e}"
            )
            if "target closed" in str(e).lower():
                current_url, page_title = "about:closed", "Page Closed"
        except Exception as e:
            logging.warning(
                f"Generic error getting page state: {type(e).__name__} - {e}"
            )
        return PageState(
            url=current_url, title=page_title, content_snippet=content_snippet
        )

    def _construct_prompt_messages(
        self, current_page_state: PageState, system_prompt_str: str
    ) -> List[BaseMessage]:
        history_str_parts = []
        for entry in self.history[-3:]:  # Last 3 entries
            history_str_parts.append(f"Step {entry.step}:")
            if entry.page_state:
                history_str_parts.append(f"  URL: {entry.page_state.url}")
            if entry.llm_response:
                history_str_parts.append(f"  Thought: {entry.llm_response.thought}")
            if entry.action_results:
                for res in entry.action_results:
                    history_str_parts.append(
                        f"  Action: {res.action_name}, Params: {res.params}, Status: {res.status}, Output: {res.output or res.error or 'N/A'}"
                    )
            if entry.step_error:
                history_str_parts.append(f"  Step Error: {entry.step_error}")

        history_summary = (
            "\n".join(history_str_parts)
            if history_str_parts
            else "No relevant history yet."
        )

        prompt_content = f"""Current Page State:
URL: {current_page_state.url}
Title: {current_page_state.title}
Content Snippet (Markdown, first 4000 chars):
{current_page_state.content_snippet}

Relevant History (condensed last 3 steps):
{history_summary}

Task: {self.task}
Based on the current state, history, and overall task, what action(s) should be taken next?
Respond ONLY with the required JSON format.
"""
        return [
            SystemMessage(content=system_prompt_str),
            HumanMessage(content=prompt_content),
        ]

    async def run(self, max_steps: int = 15):
        logging.info(f"Starting agent for task: {self.task}")
        final_task_result = "Task could not be completed within the step limit."

        action_descriptions = self.controller.get_prompt_description()
        system_prompt_str = self.system_prompt_template.format(
            task=self.task, available_actions=action_descriptions
        )

        current_step_num = 0
        for current_step_num_iter in range(1, max_steps + 1):
            current_step_num = current_step_num_iter
            logging.info(f"--- Step {current_step_num}/{max_steps} ---")

            current_page_state = None
            llm_response_model = None
            executed_action_results: List[ActionResult] = []
            current_step_error: Optional[str] = None

            try:
                current_page_state = await self._get_page_state_summary()
                if current_page_state.url == "about:closed":
                    current_step_error = (
                        "Critical: Page/Context was closed before LLM interaction."
                    )
                    logging.error(current_step_error)
                    raise PlaywrightError(current_step_error)

                messages = self._construct_prompt_messages(
                    current_page_state, system_prompt_str
                )

                logging.info(f"Invoking LLM. Current URL: {current_page_state.url}")
                llm_raw_output = await self.llm.ainvoke(messages)
                llm_output_str = str(llm_raw_output.content)

                try:
                    match = re.search(
                        r"```json\s*([\s\S]*?)\s*```|```([\s\S]*?)```", llm_output_str
                    )
                    if match:
                        llm_output_str = (
                            match.group(1) or match.group(2) or llm_output_str
                        )

                    llm_output_data = json.loads(llm_output_str.strip())
                    llm_response_model = AgentLLMResponse(**llm_output_data)
                    logging.info(f"LLM Thought: {llm_response_model.thought}")
                except (json.JSONDecodeError, ValidationError) as e_parse:
                    current_step_error = f"LLM response parsing/validation error: {type(e_parse).__name__} - {e_parse}. Raw: {llm_output_str}"
                    logging.error(current_step_error)
                    final_task_result = "Error: Could not understand AI's next actions."
                    break

                if not llm_response_model.actions:
                    current_step_error = "LLM proposed no actions."
                    logging.warning(current_step_error)
                    # Potentially add a "wait" or "noop" action if this happens often
                    # For now, we'll let it be logged in history and proceed to next step or end.

                for i, agent_action in enumerate(llm_response_model.actions or []):
                    logging.info(
                        f"Executing action {i + 1}/{len(llm_response_model.actions)}: {agent_action.action_name} with params: {agent_action.parameters}"
                    )
                    action_result = await self.controller.execute(
                        action_name=agent_action.action_name,
                        params_dict=agent_action.parameters,
                        session=self.browser_session,
                    )
                    executed_action_results.append(action_result)
                    if action_result.status == "error":
                        logging.warning(
                            f"Action '{agent_action.action_name}' failed: {action_result.error}. Stopping action sequence for this step."
                        )
                        current_step_error = (
                            action_result.error
                        )  # Record the first error
                        break
                    if action_result.is_final:
                        final_task_result = (
                            action_result.output or "Task marked as finished by agent."
                        )
                        logging.info(f"Task finished by agent: {final_task_result}")
                        # History will be logged in finally, then break main loop
                        break  # Break from actions loop

                if any(ar.is_final for ar in executed_action_results):
                    break  # Break from main steps loop

            except PlaywrightError as pwe:
                current_step_error = (
                    f"PlaywrightError during step: {type(pwe).__name__} - {pwe}"
                )
                logging.error(current_step_error)
            except Exception as e_step:
                current_step_error = f"Critical unhandled error in step {current_step_num}: {type(e_step).__name__} - {e_step}"
                logging.exception(current_step_error)
                final_task_result = (
                    current_step_error  # Task ends due to critical error
                )
                break

            finally:
                history_entry = AgentHistoryEntry(
                    step=current_step_num,
                    page_state=current_page_state,  # Could be None if error before fetch
                    llm_response=llm_response_model,
                    action_results=executed_action_results,
                    step_error=current_step_error,
                )
                self.history.append(history_entry)

        if current_step_num >= max_steps and not any(
            entry.action_results and any(ar.is_final for ar in entry.action_results)
            for entry in self.history
            if entry.action_results
        ):
            logging.warning("Max steps reached. Task not marked as finished.")
            final_task_result = "Max steps reached, task not completed."

        logging.info(
            f"\n--- Agent Run Summary ---\nTask: {self.task}\nFinal Result/Status: {final_task_result}"
        )
        return final_task_result


# --- Action Implementations ---
def setup_controller_actions(controller: Controller, llm_for_extraction: BaseChatModel):
    @controller.register_action(
        "go_to_url", "Navigates to a specific URL.", GoToUrlParams
    )
    async def go_to_url(session: BrowserSession, params: GoToUrlParams) -> ActionResult:
        page = await session.get_current_page()
        nav_timeout = session.profile_config.default_navigation_timeout
        await page.goto(
            params.url, timeout=nav_timeout, wait_until="domcontentloaded"
        )  # Consider 'load' or 'commit'
        await page.wait_for_load_state("load", timeout=nav_timeout / 2)
        return ActionResult(
            action_name="go_to_url",
            params=params.model_dump(),
            output=f"Navigated to {params.url}",
        )

    @controller.register_action(
        "click_element",
        "Clicks an element based on its description.",
        ClickElementParams,
    )
    async def click_element(
        session: BrowserSession, params: ClickElementParams
    ) -> ActionResult:
        page = await session.get_current_page()
        timeout = session.profile_config.default_timeout
        target_locator: Optional[Locator] = None
        desc_regex = re.compile(params.description, re.IGNORECASE)
        strategies = [
            (
                lambda p=page, t=params.element_type, d=desc_regex: p.get_by_role(
                    t, name=d
                )
                if t
                else None
            ),
            (lambda p=page, d=params.description: p.get_by_placeholder(d)),
            (lambda p=page, d=params.description: p.get_by_text(d, exact=False)),
            (lambda p=page, d=params.description: p.get_by_label(d)),
        ]
        for strat_func in strategies:
            try:
                loc = strat_func()
                if loc and await loc.count() > 0:
                    first_loc = loc.first
                    if await first_loc.is_visible(timeout=1500):
                        target_locator = first_loc
                        break
                    elif target_locator is None:
                        target_locator = first_loc
            except PlaywrightError:
                pass

        if not target_locator:
            return ActionResult(
                action_name="click_element",
                params=params.model_dump(),
                status="error",
                error=f"Element '{params.description}' not found.",
            )

        tag_name = await target_locator.evaluate(
            "el => el.tagName.toLowerCase()", timeout=1000
        )
        input_type = (
            (await target_locator.get_attribute("type", timeout=1000) or "").lower()
            if tag_name == "input"
            else ""
        )
        if tag_name == "input" and input_type == "file":
            return ActionResult(
                action_name="click_element",
                params=params.model_dump(),
                status="info",
                output=f"Skipped click on file input: {params.description}",
            )

        initial_pages_count = len(page.context.pages)
        url_before_click = page.url

        async def perform_click():
            await target_locator.click(timeout=timeout)  # type: ignore
            await page.wait_for_timeout(300)  # Settle

        # Handle potential new tab/page creation from click
        # Using page.context.once("page", new_page_handler) is more robust for new tabs
        new_page_event_info = asyncio.Future()

        def _new_page_handler(new_page):
            if not new_page_event_info.done():
                new_page_event_info.set_result(new_page)

        page.context.on("page", _new_page_handler)

        try:
            await perform_click()

            try:  # Check if a new page was opened
                new_page_opened = await asyncio.wait_for(
                    new_page_event_info, timeout=0.5
                )  # Short timeout
                session._current_page = new_page_opened  # Update session's current page
                await new_page_opened.bring_to_front()
                await new_page_opened.wait_for_load_state(
                    "load",
                    timeout=session.profile_config.default_navigation_timeout / 2,
                )
                output_msg = f"Clicked '{params.description}'. New tab opened and focused: {new_page_opened.url}"
            except asyncio.TimeoutError:  # No new page opened
                current_page_after_click = (
                    await session.get_current_page()
                )  # Re-fetch in case of navigation
                if current_page_after_click.url != url_before_click:
                    await current_page_after_click.wait_for_load_state(
                        "load",
                        timeout=session.profile_config.default_navigation_timeout / 2,
                    )
                    output_msg = f"Clicked '{params.description}'. Navigated to: {current_page_after_click.url}"
                else:
                    output_msg = f"Clicked element: '{params.description}'."
        finally:
            page.context.remove_listener("page", _new_page_handler)

        return ActionResult(
            action_name="click_element", params=params.model_dump(), output=output_msg
        )

    @controller.register_action(
        "input_text", "Inputs text into a described element.", InputTextParams
    )
    async def input_text(
        session: BrowserSession, params: InputTextParams
    ) -> ActionResult:
        page = await session.get_current_page()
        timeout = session.profile_config.default_timeout
        target_locator: Optional[Locator] = None
        desc_regex = re.compile(params.description, re.IGNORECASE)
        strategies = [
            (
                lambda p=page, t=params.element_type, d=desc_regex: p.get_by_role(
                    t, name=d
                )
                if t
                else None
            ),
            (lambda p=page, d=params.description: p.get_by_placeholder(d)),
            (lambda p=page, d=params.description: p.get_by_text(d, exact=False)),
            (lambda p=page, d=params.description: p.get_by_label(d)),
        ]
        for strat_func in strategies:
            try:
                loc = strat_func()
                if loc and await loc.count() > 0:
                    first_loc = loc.first
                    if await first_loc.is_visible(timeout=1500):
                        target_locator = first_loc
                        break
                    elif target_locator is None:
                        target_locator = first_loc
            except PlaywrightError:
                pass

        if not target_locator:
            return ActionResult(
                action_name="input_text",
                params=params.model_dump(),
                status="error",
                error=f"Element '{params.description}' for input not found.",
            )

        try:
            await target_locator.clear(timeout=2000)
        except PlaywrightError:
            pass
        await target_locator.fill(params.text, timeout=timeout)  # type: ignore
        return ActionResult(
            action_name="input_text",
            params=params.model_dump(),
            output=f"Typed '{params.text}' into '{params.description}'.",
        )

    @controller.register_action("scroll", "Scrolls the page up or down.", ScrollParams)
    async def scroll(session: BrowserSession, params: ScrollParams) -> ActionResult:
        page = await session.get_current_page()
        scroll_amount_js = (
            "window.innerHeight"
            if params.amount == "page"
            else str(params.amount if params.direction == "down" else -params.amount)
        )  # type: ignore
        await page.evaluate(f"window.scrollBy(0, {scroll_amount_js})")
        return ActionResult(
            action_name="scroll",
            params=params.model_dump(),
            output=f"Scrolled {params.direction} by {params.amount}.",
        )

    @controller.register_action(
        "get_content",
        "Extracts specific information from the current page based on a query.",
        GetContentParams,
    )
    async def get_content(
        session: BrowserSession, params: GetContentParams
    ) -> ActionResult:  # llm_for_extraction passed from main
        page = await session.get_current_page()
        html_content = await page.content()
        markdown_content = markdownify.markdownify(html_content)
        content_for_llm = markdown_content[:10000]

        extraction_prompt_text = f"Page Content (Markdown):\n{content_for_llm}\n\nQuery: '{params.query}'\n\nExtract ONLY the relevant information. If not found, state 'Information not found.'"

        try:
            messages = [HumanMessage(content=extraction_prompt_text)]
            response = await llm_for_extraction.ainvoke(messages)
            extracted_info = str(response.content)
        except Exception as e:
            logging.error(f"LLM extraction failed: {type(e).__name__} - {e}")
            return ActionResult(
                action_name="get_content",
                params=params.model_dump(),
                status="error",
                error=f"LLM extraction call failed: {e}",
            )

        return ActionResult(
            action_name="get_content", params=params.model_dump(), output=extracted_info
        )

    @controller.register_action(
        "wait",
        "Waits for a specified number of seconds or for a selector to become visible.",
        WaitParams,
    )
    async def wait(session: BrowserSession, params: WaitParams) -> ActionResult:
        page = await session.get_current_page()
        if params.seconds:
            await page.wait_for_timeout(params.seconds * 1000)
            return ActionResult(
                action_name="wait",
                params=params.model_dump(),
                output=f"Waited for {params.seconds} seconds.",
            )
        elif params.selector:
            await page.locator(params.selector).wait_for(
                state="visible", timeout=params.timeout_ms
            )
            return ActionResult(
                action_name="wait",
                params=params.model_dump(),
                output=f"Waited for selector '{params.selector}'.",
            )
        return ActionResult(
            action_name="wait",
            params=params.model_dump(),
            status="error",
            error="Wait action requires 'seconds' or 'selector'.",
        )

    @controller.register_action(
        "send_keys", "Sends special keyboard key presses.", SendKeysParams
    )
    async def send_keys(
        session: BrowserSession, params: SendKeysParams
    ) -> ActionResult:
        page = await session.get_current_page()
        await page.keyboard.press(params.keys)
        return ActionResult(
            action_name="send_keys",
            params=params.model_dump(),
            output=f"Pressed keys: {params.keys}",
        )

    @controller.register_action(
        "finish",
        "Indicates the task is complete and provides a final result.",
        FinishParams,
    )
    async def finish(session: BrowserSession, params: FinishParams) -> ActionResult:
        return ActionResult(
            action_name="finish",
            params=params.model_dump(),
            output=params.task_result,
            is_final=True,
        )

    # Add other actions from browser-use if needed, e.g., tab management, search_google
    # For now, keeping it to the user's existing action set.


# --- Main Execution Logic ---
async def main_async(task: str):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
    )

    persistent_profile_path = (
        "C:\\Users\\MSI\\AppData\\Local\\Google\\Chrome\\User Data\\Default"
    )
    # Example: persistent_profile_path = r"C:\Users\YourUser\AppData\Local\Google\Chrome\User Data\Default"
    chrome_exe_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    # Example: chrome_exe_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    other_browser_args = ["--start-maximized"]

    profile_config = BrowserProfileConfig(
        headless=False,
        executable_path=chrome_exe_path,
        persistent_user_data_dir=persistent_profile_path,
        browser_args=other_browser_args,
    )

    llm_instance = ChatOpenAI(
        model="gpt-4o-mini", openai_api_key=api_key, temperature=0.0
    )
    # llm_for_extraction can be the same or a different, possibly cheaper/faster model
    llm_for_extraction = ChatOpenAI(
        model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.0
    )

    action_controller = Controller()
    setup_controller_actions(
        action_controller, llm_for_extraction
    )  # Pass LLM for extraction to actions that need it

    final_output = "Agent run did not complete as expected."

    async with BrowserSession(profile_config=profile_config) as browser_session:
        agent = Agent(
            task=task,
            llm=llm_instance,
            browser_session=browser_session,
            controller=action_controller,
        )
        try:
            final_output = await agent.run(max_steps=30)
        except Exception as e:
            logging.exception("Agent run failed critically.")
            final_output = f"Agent run failed with error: {type(e).__name__}: {e}"

    logging.info(f"\nFINAL OUTPUT:\n{final_output}")


def __main__():
    default_task = "Go to google and list 5 news about covid 19 in Vietnam"
    import sys

    cli_task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    task_to_run = cli_task or default_task
    asyncio.run(main_async(task=task_to_run))
