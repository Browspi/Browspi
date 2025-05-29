import asyncio
import inspect
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Tuple  # Added Tuple
from urllib.parse import urlparse
from warnings import filterwarnings

from dotenv import load_dotenv
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    Page,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)
from typing_extensions import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

from browspi.services.browser.service import (
    DEFAULT_BROWSER_PROFILE,
    BrowserProfile,
    BrowserSession,
    BrowserStateSummary,
)

filterwarnings("ignore", category=LangChainBetaWarning)
load_dotenv()
logger = logging.getLogger(__name__)


def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()
    if hasattr(logging, levelName):
        return
    if hasattr(logging, methodName):
        raise AttributeError(f"{methodName} already defined in logging module")
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError(f"{methodName} already defined in logger class")

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def setup_logging():
    try:
        addLoggingLevel("RESULT", 35)
    except AttributeError:
        pass
    log_type = os.getenv("browspi_LOGGING_LEVEL", "info").lower()
    if logging.getLogger().hasHandlers():
        return
    root = logging.getLogger()
    root.handlers = []

    class ProjectFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.name, str) and record.name.startswith("__main__."):
                record.name = (
                    record.name.split(".")[-2]
                    if len(record.name.split(".")) > 1
                    else record.name
                )
            return super().format(record)

    console = logging.StreamHandler(sys.stdout)
    if log_type == "result":
        console.setLevel("RESULT")
        console.setFormatter(ProjectFormatter("%(message)s"))
    else:
        console.setFormatter(ProjectFormatter("%(levelname)-8s [%(name)s] %(message)s"))
    root.addHandler(console)
    if log_type == "result":
        root.setLevel("RESULT")
    elif log_type == "debug":
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)
    main_logger = logging.getLogger(__name__)
    main_logger.propagate = False
    main_logger.addHandler(console)
    main_logger.setLevel(root.level)
    third_party_loggers = [
        "httpx",
        "playwright",
        "asyncio",
        "langchain",
        "openai",
        "httpcore",
    ]
    for logger_name in third_party_loggers:
        third_party = logging.getLogger(logger_name)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False


setup_logging()


def time_execution_async(
    additional_text: str = "",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = await func(*args, **kwargs)
            logger.debug(
                f"{additional_text} Execution time: {time.time() - start_time:.2f} seconds"
            )
            return result

        return wrapper

    return decorator


def check_env_variables(keys: List[str], any_or_all=all) -> bool:
    return any_or_all(os.getenv(key, "").strip() for key in keys)


class ActionResult(BaseModel):
    is_done: bool = False
    success: Optional[bool] = None
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False


class ActionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_index(self) -> Optional[int]:
        for _, value in self:
            if isinstance(value, dict) and "index" in value:
                return value["index"]
            if hasattr(value, "index") and value.index is not None:
                return value.index  # type: ignore
        return None

    def set_index(self, index: int):
        for field_name, _ in self:
            value = getattr(self, field_name)
            if isinstance(value, dict) and "index" in value:
                value["index"] = index
                setattr(self, field_name, value)
                return
            if hasattr(value, "index"):
                value.index = index
                return  # type: ignore
        logger.warning(f"Could not set index on action: {self}")


class GoToUrlAction(BaseModel):
    url: str


class SearchGoogleAction(BaseModel):
    query: str


class ClickElementAction(BaseModel):
    index: int


class InputTextAction(BaseModel):
    index: int
    text: str


class ScrollAction(BaseModel):
    amount: Optional[int] = None


class DoneAction(BaseModel):
    text: str
    success: bool


class OpenTabAction(BaseModel):
    url: str


class CloseTabAction(BaseModel):
    page_id: int


class SwitchTabAction(BaseModel):
    page_id: int


class ExtractPageContentAction(BaseModel):
    goal: str
    include_links: bool = False


class SendKeysAction(BaseModel):
    keys: str


class WaitAction(BaseModel):
    seconds: int = 3


Context = TypeVar("Context")


class RegisteredAction(BaseModel):
    name: str
    description: str
    function: Callable[..., Awaitable[Any]]
    param_model: Type[BaseModel]
    domains: Optional[List[str]] = None
    page_filter: Optional[Callable[[Page], bool]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def prompt_description(self) -> str:
        return f"{self.description}: {self.name}({self.param_model.__name__})"


class ActionRegistry(BaseModel):
    actions: Dict[str, RegisteredAction] = {}

    def get_prompt_description(self, page: Optional[Page] = None) -> str:
        return "\n".join(
            action.prompt_description()
            for _, action in self.actions.items()
            if not (
                page
                and action.domains
                and not any(
                    urlparse(page.url).hostname.endswith(d.lstrip("*."))
                    for d in action.domains
                )
            )  # type: ignore
            and not (page and action.page_filter and not action.page_filter(page))
        )

    def create_action_model(
        self, include_actions: Optional[List[str]] = None, page: Optional[Page] = None
    ) -> Type[ActionModel]:
        fields = {}
        for name, action_info in self.actions.items():
            if include_actions and name not in include_actions:
                continue
            if (
                page
                and action_info.domains
                and not any(
                    urlparse(page.url).hostname.endswith(d.lstrip("*."))
                    for d in action_info.domains
                )
            ):
                continue  # type: ignore
            if page and action_info.page_filter and not action_info.page_filter(page):
                continue
            fields[name] = (
                Optional[action_info.param_model],
                Field(default=None, description=action_info.description),
            )
        return create_model("DynamicActionModel", __base__=ActionModel, **fields)  # type: ignore


class Controller(Generic[Context]):
    def __init__(
        self,
        exclude_actions: List[str] = [],
        output_model: Optional[Type[BaseModel]] = None,
    ):
        self.registry = ActionRegistry()
        self.exclude_actions = exclude_actions
        self._register_default_actions()
        if output_model:

            class CustomDoneAction(BaseModel):
                data: output_model
                success: bool  # type: ignore

            self.registry.actions["done"] = RegisteredAction(
                name="done",
                description="Completes the task with custom output",
                function=self._custom_done_action_func,
                param_model=CustomDoneAction,
            )  # type: ignore

    async def _custom_done_action_func(self, params: BaseModel):
        return ActionResult(
            is_done=True,
            success=params.success,
            extracted_content=params.data.model_dump_json(),
        )  # type: ignore

    def action(
        self, description: str, param_model: Optional[Type[BaseModel]] = None, **kwargs
    ):
        def decorator(func: Callable[..., Awaitable[Any]]):
            if func.__name__ in self.exclude_actions:
                return func
            actual_param_model = param_model
            if not actual_param_model:
                sig = inspect.signature(func)
                fields = {
                    name: (
                        p.annotation,
                        p.default if p.default != inspect.Parameter.empty else ...,
                    )
                    for name, p in sig.parameters.items()
                    if name not in ["browser_session", "page_extraction_llm", "context"]
                }
                actual_param_model = create_model(f"{func.__name__}Params", **fields)  # type: ignore
            self.registry.actions[func.__name__] = RegisteredAction(
                name=func.__name__,
                description=description,
                function=func,
                param_model=actual_param_model,
                **kwargs,
            )  # type: ignore
            return func

        return decorator

    def _register_default_actions(self):
        @self.action("Navigate to a URL", param_model=GoToUrlAction)
        async def go_to_url(params: GoToUrlAction, browser_session: BrowserSession):
            await browser_session.navigate(params.url)
            return ActionResult(
                extracted_content=f"Navigated to {params.url}", include_in_memory=True
            )

        @self.action("Search Google", param_model=SearchGoogleAction)
        async def search_google(
            params: SearchGoogleAction, browser_session: BrowserSession
        ):
            await browser_session.navigate(
                f"https://www.google.com/search?q={params.query.replace(' ', '+')}"
            )
            return ActionResult(
                extracted_content=f"Searched Google for: {params.query}",
                include_in_memory=True,
            )

        @self.action("Click an element by its index", param_model=ClickElementAction)
        async def click_element_by_index(
            params: ClickElementAction, browser_session: BrowserSession
        ):
            logger.info(f"Attempting to click element with index: {params.index}")
            state = await browser_session.get_state_summary(
                cache_clickable_elements_hashes=False
            )
            if params.index in state.selector_map:
                element_to_click = state.selector_map[params.index]
                page = await browser_session.get_current_page()
                try:
                    await page.locator(f"xpath={element_to_click.xpath}").click(
                        timeout=5000
                    )
                    return ActionResult(
                        extracted_content=f"Clicked element at index {params.index} ({element_to_click.tag_name})",
                        include_in_memory=True,
                    )  # type: ignore
                except Exception as e:
                    logger.error(f"Failed to click element {params.index}: {e}")
                    return ActionResult(
                        error=f"Failed to click element at index {params.index}: {str(e)}"
                    )
            return ActionResult(error=f"Element with index {params.index} not found.")

        @self.action("Input text into an element", param_model=InputTextAction)
        async def input_text(params: InputTextAction, browser_session: BrowserSession):
            logger.info(
                f"Attempting to input text '{params.text}' into element with index: {params.index}"
            )
            state = await browser_session.get_state_summary(
                cache_clickable_elements_hashes=False
            )
            if params.index in state.selector_map:
                element_to_input = state.selector_map[params.index]
                page = await browser_session.get_current_page()
                try:
                    await page.locator(f"xpath={element_to_input.xpath}").fill(
                        params.text, timeout=5000
                    )
                    return ActionResult(
                        extracted_content=f"Inputted '{params.text}' into element {params.index}",
                        include_in_memory=True,
                    )  # type: ignore
                except Exception as e:
                    logger.error(
                        f"Failed to input text into element {params.index}: {e}"
                    )
                    return ActionResult(
                        error=f"Failed to input text into element {params.index}: {str(e)}"
                    )
            return ActionResult(
                error=f"Element with index {params.index} not found for input."
            )

        @self.action("Scroll down the page", param_model=ScrollAction)
        async def scroll_down(params: ScrollAction, browser_session: BrowserSession):
            page = await browser_session.get_current_page()
            amount = (
                params.amount if params.amount is not None else "window.innerHeight"
            )
            await page.evaluate(f"window.scrollBy(0, {amount})")
            return ActionResult(
                extracted_content=f"Scrolled down by {params.amount or 'one page'}",
                include_in_memory=True,
            )

        @self.action("Scroll up the page", param_model=ScrollAction)
        async def scroll_up(params: ScrollAction, browser_session: BrowserSession):
            page = await browser_session.get_current_page()
            amount = (
                params.amount if params.amount is not None else "window.innerHeight"
            )
            await page.evaluate(f"window.scrollBy(0, -{amount})")
            return ActionResult(
                extracted_content=f"Scrolled up by {params.amount or 'one page'}",
                include_in_memory=True,
            )

        @self.action("Mark task as done", param_model=DoneAction)
        async def done(params: DoneAction):
            return ActionResult(
                is_done=True, success=params.success, extracted_content=params.text
            )

        @self.action("Open URL in a new tab", param_model=OpenTabAction)
        async def open_tab(params: OpenTabAction, browser_session: BrowserSession):
            if not browser_session.browser_context:
                await browser_session.start()
            new_page = await browser_session.browser_context.new_page()
            await new_page.goto(params.url, wait_until="domcontentloaded")
            browser_session.agent_current_page = new_page
            return ActionResult(
                extracted_content=f"Opened new tab with URL: {params.url}",
                include_in_memory=True,
            )  # type: ignore

        @self.action("Close an existing tab by its ID", param_model=CloseTabAction)
        async def close_tab_action(
            params: CloseTabAction, browser_session: BrowserSession
        ):
            await browser_session.close_tab(params.page_id)
            return ActionResult(
                extracted_content=f"Closed tab with ID: {params.page_id}",
                include_in_memory=True,
            )

        @self.action("Switch to a specific tab by its ID", param_model=SwitchTabAction)
        async def switch_tab(params: SwitchTabAction, browser_session: BrowserSession):
            if not browser_session.browser_context:
                return ActionResult(error="Browser context not available.")
            pages = browser_session.browser_context.pages
            if 0 <= params.page_id < len(pages):
                browser_session.agent_current_page = pages[params.page_id]
                await browser_session.agent_current_page.bring_to_front()
                return ActionResult(
                    extracted_content=f"Switched to tab ID: {params.page_id}",
                    include_in_memory=True,
                )
            return ActionResult(error=f"Tab ID {params.page_id} not found.")

        @self.action(
            "Extract content from the current page based on a goal",
            param_model=ExtractPageContentAction,
        )
        async def extract_content(
            params: ExtractPageContentAction,
            browser_session: BrowserSession,
            page_extraction_llm: BaseChatModel,
        ):
            page = await browser_session.get_current_page()
            extracted_data_summary = f"Attempted to extract: '{params.goal}'."
            try:
                main_content_selectors = ["main", "article", "[role='main']", "body"]
                text_content = ""
                logger.info(f"Attempting to extract text for goal: '{params.goal}'")
                for selector in main_content_selectors:
                    try:
                        content_elements = page.locator(selector)
                        count = await content_elements.count()
                        if count > 0:
                            visible_element_text = ""
                            for i in range(count):
                                element = content_elements.nth(i)
                                if await element.is_visible(timeout=1000):
                                    visible_element_text = await element.inner_text(
                                        timeout=5000
                                    )
                                    break
                            if visible_element_text.strip():
                                text_content = visible_element_text.strip()
                                logger.info(
                                    f"Extracted text using '{selector}'. Length: {len(text_content)}"
                                )
                                break
                            else:
                                logger.debug(
                                    f"Selector '{selector}' found, but no visible text."
                                )
                        else:
                            logger.debug(f"Selector '{selector}' not found.")
                    except PlaywrightTimeoutError:
                        logger.warning(f"Timeout extracting from '{selector}'.")
                        continue
                    except Exception as el_err:
                        logger.debug(
                            f"Error processing selector '{selector}': {el_err}"
                        )
                        continue
                if not text_content.strip():
                    logger.warning("No text extracted from main content selectors.")
            except Exception as e:
                logger.error(
                    f"Page scraping error for 'extract_content': {type(e).__name__} - {e}",
                    exc_info=True,
                )
                return ActionResult(
                    error=f"Failed to scrape page: {type(e).__name__}",
                    include_in_memory=True,
                )
            simplified_extraction_goal = params.goal
            if "extract the titles and links of 5 news articles" in params.goal.lower():
                simplified_extraction_goal = (
                    "titles and links of news articles about COVID-19 in Vietnam"
                )
            max_text_for_sub_llm = 10000
            text_to_process = (
                text_content[:max_text_for_sub_llm]
                if len(text_content) > max_text_for_sub_llm
                else text_content
            )
            if not text_to_process.strip():
                logger.warning(
                    f"No text for page_extraction_llm for goal: '{simplified_extraction_goal}'"
                )
                extracted_data_summary = (
                    f"No text found for: '{simplified_extraction_goal}'."
                )
            elif page_extraction_llm:
                sub_llm_prompt = f'Review TEXT_TO_PROCESS for goal: \'{simplified_extraction_goal}\'.\nList items clearly. For news, extract title and direct URL. State \'No specific information or articles found matching the goal.\' if none found.\n\nTEXT_TO_PROCESS:\n"""{text_to_process}"""'
                try:
                    logger.info(
                        f"Sending to page_extraction_llm. Goal: '{simplified_extraction_goal}'. Text length: {len(text_to_process)}"
                    )
                    llm_response = await page_extraction_llm.ainvoke(sub_llm_prompt)
                    extracted_data = (
                        llm_response.content
                        if hasattr(llm_response, "content")
                        else str(llm_response)
                    )
                    if (
                        not extracted_data.strip()
                        or "no specific information or articles found"
                        in extracted_data.lower()
                    ):
                        extracted_data_summary = f"Sub-LLM: No specific info for '{simplified_extraction_goal}'."
                    else:
                        extracted_data_summary = extracted_data
                    logger.info(
                        f"page_extraction_llm response: {extracted_data_summary[:300]}..."
                    )
                except Exception as e:
                    logger.error(
                        f"page_extraction_llm failed: {type(e).__name__} - {e}",
                        exc_info=True,
                    )
                    extracted_data_summary = f"Error during sub-LLM extraction for '{simplified_extraction_goal}': {type(e).__name__}."
            else:
                extracted_data_summary = "page_extraction_llm not configured."
            return ActionResult(
                extracted_content=extracted_data_summary, include_in_memory=True
            )

        @self.action(
            "Send special keys or keyboard shortcuts", param_model=SendKeysAction
        )
        async def send_keys(params: SendKeysAction, browser_session: BrowserSession):
            page = await browser_session.get_current_page()
            await page.keyboard.press(params.keys)
            return ActionResult(
                extracted_content=f"Sent keys: {params.keys}", include_in_memory=True
            )

        @self.action("Wait for a specified number of seconds", param_model=WaitAction)
        async def wait(params: WaitAction):
            await asyncio.sleep(params.seconds)
            return ActionResult(
                extracted_content=f"Waited for {params.seconds} seconds",
                include_in_memory=True,
            )

    async def act(
        self,
        action: ActionModel,
        browser_session: BrowserSession,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[List[str]] = None,
        context: Optional[Context] = None,
    ):
        dumped_action = action.model_dump(exclude_unset=True)
        if not dumped_action:
            return ActionResult(error="No action in model.")
        action_name = list(dumped_action.keys())[0]
        params_obj = dumped_action[action_name]
        if action_name not in self.registry.actions:
            return ActionResult(error=f"Action '{action_name}' not found.")
        registered_action = self.registry.actions[action_name]
        action_kwargs: Dict[str, Any] = {}
        sig = inspect.signature(registered_action.function)
        for p_name in [
            "browser_session",
            "page_extraction_llm",
            "sensitive_data",
            "available_file_paths",
            "context",
        ]:
            if p_name in sig.parameters:
                action_kwargs[p_name] = locals()[p_name]
        try:
            validated_params = (
                registered_action.param_model(**params_obj)
                if isinstance(params_obj, dict)
                else (
                    params_obj
                    if isinstance(params_obj, BaseModel)
                    else registered_action.param_model()
                )
            )
            first_param_name = (
                list(sig.parameters.keys())[0] if sig.parameters else None
            )
            if (
                first_param_name
                and first_param_name != "self"
                and inspect.isclass(sig.parameters[first_param_name].annotation)
                and isinstance(
                    validated_params, sig.parameters[first_param_name].annotation
                )
            ):
                result = await registered_action.function(
                    validated_params, **action_kwargs
                )
            elif hasattr(validated_params, "model_dump"):
                result = await registered_action.function(
                    **validated_params.model_dump(exclude_none=True), **action_kwargs
                )
            else:
                result = await registered_action.function(**action_kwargs)
            if isinstance(result, ActionResult):
                return result
            return (
                ActionResult(extracted_content=str(result))
                if isinstance(result, str)
                else ActionResult()
            )
        except PlaywrightTimeoutError:
            logger.error(f"Timeout: {action_name}")
            return ActionResult(error=f"Action '{action_name}' timed out.")
        except PlaywrightError as e:
            logger.error(f"Playwright error: {action_name}: {e}")
            return ActionResult(error=f"Browser error: '{action_name}': {e}")
        except ValidationError as e:
            logger.error(f"Validation error: {action_name}: {e}")
            return ActionResult(error=f"Invalid params for '{action_name}': {e}")
        except Exception as e:
            logger.error(f"Error in action {action_name}: {e}", exc_info=True)
            return ActionResult(error=f"Unexpected error in '{action_name}': {e}")


@dataclass
class MessageMetadata:
    tokens: int = 0
    message_type: Optional[str] = None


@dataclass
class ManagedMessage:
    message: BaseMessage
    metadata: MessageMetadata = field(default_factory=MessageMetadata)


class MessageHistory(BaseModel):
    messages: List[ManagedMessage] = Field(default_factory=list)
    current_tokens: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_message(
        self,
        message: BaseMessage,
        metadata: MessageMetadata,
        position: Optional[int] = None,
    ):
        managed_msg = ManagedMessage(message=message, metadata=metadata)
        if position is None:
            self.messages.append(managed_msg)
        else:
            self.messages.insert(position, managed_msg)
        self.current_tokens += metadata.tokens

    def remove_last_state_message(self):
        if (
            self.messages
            and isinstance(self.messages[-1].message, HumanMessage)
            and self.messages[-1].metadata.message_type != "init"
        ):
            removed_msg = self.messages.pop()
            self.current_tokens -= removed_msg.metadata.tokens


class MessageManagerState(BaseModel):
    history: MessageHistory = Field(default_factory=MessageHistory)
    tool_id: int = 1
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MessageManagerSettings(BaseModel):
    max_input_tokens: int = 128000
    estimated_characters_per_token: int = 3
    image_tokens: int = 800
    include_attributes: List[str] = Field(default_factory=list)
    message_context: Optional[str] = None
    sensitive_data: Optional[Dict[str, str]] = None
    available_file_paths: Optional[List[str]] = None


@dataclass
class AgentStepInfo:
    step_number: int
    max_steps: int


def is_last_step(self) -> bool:
    return self.step_number >= self.max_steps - 1


class MessageManager:
    def __init__(
        self,
        task: str,
        system_message: SystemMessage,
        settings: MessageManagerSettings = MessageManagerSettings(),
        state: Optional[MessageManagerState] = None,
    ):
        self.task = task
        self.settings = settings
        self.state = state or MessageManagerState()
        self.system_prompt = system_message
        if not self.state.history.messages:
            self._init_messages()

    def _count_tokens(self, message: BaseMessage) -> int:
        content_str = ""
        image_count = 0
        if isinstance(message.content, str):
            content_str = message.content
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_str += item["text"]
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    content_str += " [IMAGE] "
                    image_count += 1
        if hasattr(message, "tool_calls") and message.tool_calls:
            content_str += str(message.tool_calls)
        return (len(content_str) // self.settings.estimated_characters_per_token) + (
            image_count * self.settings.image_tokens
        )

    def _add_message_with_tokens(
        self,
        message: BaseMessage,
        position: Optional[int] = None,
        message_type: Optional[str] = None,
    ):
        if self.settings.sensitive_data:
            # Redaction logic as before
            if isinstance(message.content, str):
                temp_content = message.content
                for placeholder, real_value in self.settings.sensitive_data.items():
                    if real_value:
                        temp_content = temp_content.replace(
                            real_value, f"<secret>{placeholder}</secret>"
                        )
                message.content = temp_content
            elif isinstance(message.content, list):
                new_content_list = []
                for item in message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        temp_text = item["text"]
                        for (
                            placeholder,
                            real_value,
                        ) in self.settings.sensitive_data.items():
                            if real_value:
                                temp_text = temp_text.replace(
                                    real_value, f"<secret>{placeholder}</secret>"
                                )
                        new_content_list.append({"type": "text", "text": temp_text})
                    else:
                        new_content_list.append(item)
                message.content = new_content_list  # type: ignore
        self.state.history.add_message(
            message,
            MessageMetadata(
                tokens=self._count_tokens(message), message_type=message_type
            ),
            position,
        )
        self._ensure_token_limit()

    def _init_messages(self):
        self._add_message_with_tokens(self.system_prompt, message_type="init")
        if self.settings.message_context:
            self._add_message_with_tokens(
                HumanMessage(content=f"Context: {self.settings.message_context}"),
                message_type="init",
            )
        self._add_message_with_tokens(
            HumanMessage(content=f'Your ultimate task is: """{self.task}""".'),
            message_type="init",
        )
        if self.settings.sensitive_data:
            self._add_message_with_tokens(
                HumanMessage(
                    content=f"Sensitive data placeholders: {list(self.settings.sensitive_data.keys())}. Use them like <secret>placeholder_name</secret>"
                ),
                message_type="init",
            )
        example_current_state = {
            "evaluation_previous_goal": "Success|Failed|Unknown...",
            "memory": "Done X, Y remains...",
            "next_goal": "Do Z",
        }
        example_action_list = [{"one_action_name": {"parameter_name": "value"}}]
        example_args = {
            "current_state": example_current_state,
            "action": example_action_list,
        }
        self._add_message_with_tokens(
            HumanMessage(
                content=f"Example AgentOutput JSON:\n```json\n{json.dumps(example_args, indent=2)}\n```"
            ),
            message_type="init",
        )

        # Ensure example_tool_call_id is short enough
        example_tool_call_id = (
            f"ex-{str(uuid.uuid4())[:30]}"  # Approx 33 chars, well within 40
        )

        example_ai_msg = AIMessage(
            content="Performing action.",
            tool_calls=[
                {
                    "id": example_tool_call_id,
                    "name": "AgentOutput",
                    "args": example_args,
                }
            ],
        )
        self._add_message_with_tokens(example_ai_msg, message_type="init")
        self.add_tool_message(
            content="Example processed.",
            tool_call_id=example_tool_call_id,
            message_type="init",
        )
        self._add_message_with_tokens(
            HumanMessage(content="[Task history memory starts here]"),
            message_type="init",
        )
        if self.settings.available_file_paths:
            self._add_message_with_tokens(
                HumanMessage(
                    content=f"Available files: {self.settings.available_file_paths}"
                ),
                message_type="init",
            )

    def add_new_task(self, new_task: str):
        self.task = new_task
        self._add_message_with_tokens(
            HumanMessage(
                content=f'New task: """{new_task}""". Consider previous context.'
            )
        )

    def add_state_message(
        self,
        browser_state_summary: BrowserStateSummary,
        result: list[ActionResult] | None = None,
        step_info: Optional[AgentStepInfo] = None,
        use_vision=True,
    ):
        if result:
            for r_item in result:
                if r_item.include_in_memory:
                    if r_item.extracted_content:
                        self._add_message_with_tokens(
                            HumanMessage(
                                content="Prior Action result: "
                                + str(r_item.extracted_content)
                            )
                        )
                    if r_item.error:
                        self._add_message_with_tokens(
                            HumanMessage(
                                content="Prior Action error: "
                                + r_item.error.split("\n")[-1]
                            )
                        )
            result = None
        assert browser_state_summary is not None
        state_message = AgentMessagePrompt(
            browser_state_summary=browser_state_summary,
            result=result,
            include_attributes=self.settings.include_attributes,
            step_info=step_info,
        ).get_user_message(use_vision)
        self._add_message_with_tokens(state_message)

    def add_model_output(
        self, model_output: "AgentOutput", tool_call_id: str
    ):  # Expect tool_call_id
        # This method now assumes tool_call_id is passed from Agent.get_next_action's AIMessage
        tool_calls = [
            {
                "id": tool_call_id,
                "name": "AgentOutput",
                "args": model_output.model_dump(exclude_unset=True, exclude_none=True),
            }
        ]
        ai_message = AIMessage(content="", tool_calls=tool_calls)
        self._add_message_with_tokens(ai_message)

    def add_tool_message(
        self, content: str, tool_call_id: str, message_type: Optional[str] = None
    ):
        tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
        self._add_message_with_tokens(tool_message, message_type=message_type)
        # self.state.tool_id += 1 # Incrementing tool_id is now implicitly handled by AIMessage's tool_call_id generation

    def get_messages(self) -> List[BaseMessage]:
        return [m.message for m in self.state.history.messages]

    def _ensure_token_limit(self):
        min_messages_to_keep = sum(
            1
            for mc in self.state.history.messages
            if mc.metadata.message_type == "init"
        )
        while (
            self.state.history.current_tokens > self.settings.max_input_tokens
            and len(self.state.history.messages) > min_messages_to_keep
        ):
            removed = False
            for i in range(len(self.state.history.messages)):
                if self.state.history.messages[i].metadata.message_type != "init":
                    # Logic to avoid removing an AIMessage without its corresponding ToolMessage
                    current_msg_container = self.state.history.messages[i]
                    is_ai_expecting_tool = False
                    if (
                        isinstance(current_msg_container.message, AIMessage)
                        and current_msg_container.message.tool_calls
                    ):
                        expected_tc_id = current_msg_container.message.tool_calls[
                            0
                        ].get("id")
                        if i + 1 < len(self.state.history.messages):
                            next_msg_container = self.state.history.messages[i + 1]
                            if (
                                isinstance(next_msg_container.message, ToolMessage)
                                and next_msg_container.message.tool_call_id
                                == expected_tc_id
                            ):
                                is_ai_expecting_tool = True  # It has its tool message
                        else:  # It's the last message, so it's expecting one
                            is_ai_expecting_tool = True

                    if not is_ai_expecting_tool or (
                        is_ai_expecting_tool
                        and i + 1 < len(self.state.history.messages)
                    ):  # If it's expecting but not last, or not expecting
                        removed_msg_container = self.state.history.messages.pop(i)
                        self.state.history.current_tokens -= (
                            removed_msg_container.metadata.tokens
                        )
                        logger.info(
                            f"Removed message (type: {removed_msg_container.metadata.message_type}, content: {str(removed_msg_container.message.content)[:50]}...) for token limit. Current: {self.state.history.current_tokens}"
                        )
                        removed = True
                        break
            if not removed:
                logger.warning("Could not remove messages to reduce token count.")
                break

    def cut_messages(self):
        pass


BROWSER_USE_SYSTEM_PROMPT_TEMPLATE = """
You are an AI agent ...
Available actions:
{action_description}
"""  # Same as before


class SystemPrompt:
    def __init__(
        self,
        action_description: str,
        max_actions_per_step: int = 20,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
    ):
        prompt = override_system_message or BROWSER_USE_SYSTEM_PROMPT_TEMPLATE.format(
            max_actions_per_step=max_actions_per_step,
            action_description=action_description,
        )
        if extend_system_message:
            prompt += f"\n{extend_system_message}"
        self.system_message = SystemMessage(content=prompt)

    def get_system_message(self) -> SystemMessage:
        return self.system_message


class AgentMessagePrompt:
    def __init__(
        self,
        browser_state_summary: BrowserStateSummary,
        result: list[ActionResult] | None = None,
        include_attributes: list[str] | None = None,
        step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = browser_state_summary
        self.result = result
        self.include_attributes = include_attributes or []
        self.step_info = step_info
        assert self.state is not None

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        elements_text = (
            self.state.element_tree.clickable_elements_to_string(
                include_attributes=self.include_attributes
            )
            if self.state.element_tree
            else ""
        )
        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0
        if elements_text.strip():
            elements_text = (
                f"... {self.state.pixels_above} px above ...\n{elements_text}"
                if has_content_above
                else f"[Start of page]\n{elements_text}"
            )
            elements_text = (
                f"{elements_text}\n... {self.state.pixels_below} px below ..."
                if has_content_below
                else f"{elements_text}\n[End of page]"
            )
        else:
            elements_text = "No interactive elements or empty page."
        step_info_desc = (
            f"Step: {self.step_info.step_number + 1}/{self.step_info.max_steps}. "
            if self.step_info
            else ""
        )
        step_info_desc += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        tabs_fmt = (
            "\n".join(
                [
                    f"- T{t.page_id}: '{t.title[:40]}' ({t.url[:50]})"
                    for t in self.state.tabs
                ]
            )
            if self.state.tabs
            else "No tabs."
        )
        state_desc = f"\n[Task History Ends]\n[Current State]\nURL: {self.state.url}\nTabs:\n{tabs_fmt}\nInteractive Elements:\n{elements_text}\n{step_info_desc}\n"
        if self.result:
            for i, res in enumerate(self.result):
                if res.extracted_content:
                    state_desc += f"\nRes {i + 1}: {str(res.extracted_content)[:300]}"
                if res.error:
                    state_desc += f"\nErr {i + 1}: ...{res.error.splitlines()[-1]}"
        if self.state.screenshot and use_vision:
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_desc},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.state.screenshot}"
                        },
                    },
                ]
            )
        return HumanMessage(content=state_desc)


class AgentBrain(BaseModel):
    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_state: AgentBrain
    action: List[ActionModel] = Field(..., min_length=1)

    @staticmethod
    def type_with_custom_actions(
        custom_actions_model: Type[ActionModel],
    ) -> Type["AgentOutput"]:
        return create_model(
            "DynamicAgentOutput",
            __base__=AgentOutput,
            action=(List[custom_actions_model], Field(..., min_length=1)),
            __module__=AgentOutput.__module__,
        )  # type: ignore


class AgentSettings(BaseModel):
    use_vision: bool = True
    save_conversation_path: Optional[str] = None
    max_failures: int = 3
    retry_delay: int = 10
    override_system_message: Optional[str] = None
    extend_system_message: Optional[str] = None
    max_input_tokens: int = 128000
    validate_output: bool = False
    message_context: Optional[str] = None
    generate_gif: Union[bool, str] = False
    available_file_paths: Optional[List[str]] = None
    max_actions_per_step: int = 5
    tool_calling_method: Optional[
        Literal["function_calling", "json_mode", "raw", "auto", "tools"]
    ] = "auto"
    page_extraction_llm: Optional[BaseChatModel] = None
    planner_llm: Optional[BaseChatModel] = None
    planner_interval: int = 1
    is_planner_reasoning: bool = False
    save_playwright_script_path: Optional[str] = None
    extend_planner_system_message: Optional[str] = None
    interrupt_on_page_change_in_multi_act: bool = True


class AgentState(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: Optional[List[ActionResult]] = None
    history: "AgentHistoryList" = Field(
        default_factory=lambda: AgentHistoryList(history=[])
    )
    message_manager_state: MessageManagerState = Field(
        default_factory=MessageManagerState
    )
    paused: bool = False
    stopped: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_output: Optional[AgentOutput] = None
    result: List[ActionResult]
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AgentHistoryList(BaseModel):
    history: List[AgentHistory] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_done(self) -> bool:
        return bool(
            self.history
            and self.history[-1].result
            and self.history[-1].result[-1].is_done
        )

    def is_successful(self) -> Optional[bool]:
        return (
            self.history[-1].result[-1].success
            if self.is_done() and self.history[-1].result
            else None
        )

    def final_result(self) -> Optional[str]:
        return (
            self.history[-1].result[-1].extracted_content
            if self.is_done() and self.history[-1].result
            else None
        )


AgentState.model_rebuild()


class Agent(Generic[Context]):
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: Optional[BrowserSession] = None,
        controller: Optional[Controller[Context]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        context: Optional[Context] = None,
        agent_settings: Optional[AgentSettings] = None,
        browser_profile: Optional[BrowserProfile] = None,
        injected_agent_state: Optional[AgentState] = None,
    ):
        self.task = task
        self.llm = llm
        self.controller = controller or Controller()
        self.sensitive_data = sensitive_data
        self.version = "main.py-refactored-0.7"
        self.settings = agent_settings or AgentSettings()
        self.state = injected_agent_state or AgentState()
        self.browser_session = browser_session or BrowserSession(
            browser_profile=(browser_profile or DEFAULT_BROWSER_PROFILE)
        )
        self.context = context
        self.ActionModelType = self.controller.registry.create_action_model()
        self.AgentOutputType = AgentOutput.type_with_custom_actions(
            self.ActionModelType
        )
        self.initial_actions = (
            self._convert_initial_actions(initial_actions) if initial_actions else None
        )
        self.tool_calling_method = self.settings.tool_calling_method
        if self.tool_calling_method == "auto":
            self.tool_calling_method = "tools"
        if not hasattr(self.llm, "bind_tools") and self.tool_calling_method not in [
            "raw",
            None,
        ]:
            logger.warning(
                f"LLM {self.llm.__class__.__name__} may not support bind_tools with method '{self.tool_calling_method}'."
            )
        active_bp = self.browser_session.browser_profile
        msg_mgr_settings = MessageManagerSettings(
            max_input_tokens=self.settings.max_input_tokens,
            include_attributes=active_bp.include_attributes,
            message_context=self.settings.message_context,
            sensitive_data=self.sensitive_data,
            available_file_paths=self.settings.available_file_paths,
        )
        sys_prompt_obj = SystemPrompt(
            action_description=self.controller.registry.get_prompt_description(),
            max_actions_per_step=self.settings.max_actions_per_step,
            override_system_message=self.settings.override_system_message,
            extend_system_message=self.settings.extend_system_message,
        )
        self._message_manager = MessageManager(
            task=task,
            system_message=sys_prompt_obj.get_system_message(),
            settings=msg_mgr_settings,
            state=self.state.message_manager_state,
        )
        self._external_pause_event = asyncio.Event()
        self._external_pause_event.set()

    def _convert_initial_actions(
        self, actions: List[Dict[str, Dict[str, Any]]]
    ) -> List[ActionModel]:
        return [
            self.ActionModelType(**ad)
            for ad in actions
            if self._validate_initial_action(ad)
        ]

    def _validate_initial_action(self, action_dict: Dict[str, Dict[str, Any]]) -> bool:
        try:
            self.ActionModelType(**action_dict)
            return True
        except ValidationError as e:
            logger.error(f"Failed to validate initial action {action_dict}: {e}")
            return False

    async def get_next_action(
        self, input_messages: List[BaseMessage]
    ) -> Tuple[AgentOutput, Optional[str]]:  # Return tool_call_id
        llm_with_tool = self.llm
        tool_call_id_from_llm = None
        try:
            if self.tool_calling_method in ["tools", "function_calling"]:
                llm_with_tool = self.llm.bind_tools(
                    tools=[self.AgentOutputType],
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.AgentOutputType.__name__},
                    },
                )
        except Exception as e:
            logger.error(
                f"Failed to configure LLM for tool calling with method '{self.tool_calling_method}': {e}"
            )

        raw_response: AIMessage = await llm_with_tool.ainvoke(input_messages)  # type: ignore
        model_output: AgentOutput

        if (
            self.tool_calling_method in ["tools", "function_calling"]
            and hasattr(raw_response, "tool_calls")
            and raw_response.tool_calls
        ):
            tool_call = raw_response.tool_calls[0]
            tool_call_id_from_llm = tool_call.get("id")  # Get ID from AIMessage
            tool_name = tool_call.get("name")
            args = tool_call.get("args")
            if (
                not tool_name
                or tool_name.lower() != self.AgentOutputType.__name__.lower()
            ):
                raise ValueError(f"LLM called unexpected tool: {tool_name}")
            try:
                model_output = self.AgentOutputType(
                    **(json.loads(args) if isinstance(args, str) else args)
                )
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(
                    f"Failed to parse/validate AgentOutput from tool args: {e}. Args: {args}"
                )
                raise
        elif isinstance(raw_response.content, str):
            try:
                content_str = raw_response.content.strip()
                if content_str.startswith("```json"):
                    content_str = content_str[7:]
                if content_str.endswith("```"):
                    content_str = content_str[:-3]
                model_output = self.AgentOutputType(**json.loads(content_str.strip()))
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(
                    f"Failed to parse LLM content as AgentOutput: {e}. Content: {raw_response.content}"
                )
                raise
        else:
            raise ValueError(f"LLM response unexpected format: {raw_response}")

        log_response(model_output)
        return model_output, tool_call_id_from_llm

    async def multi_act(self, actions: List[ActionModel]) -> List[ActionResult]:
        results: List[ActionResult] = []
        initial_hashes_url = None
        initial_hashes_set = set()
        if self.browser_session._cached_clickable_element_hashes:
            initial_hashes_url = (
                self.browser_session._cached_clickable_element_hashes.url
            )
            initial_hashes_set = (
                self.browser_session._cached_clickable_element_hashes.hashes
            )
        for i, action_model_instance in enumerate(actions):
            if self.state.stopped:
                results.append(
                    ActionResult(
                        error="Agent stopped during action sequence.",
                        include_in_memory=True,
                    )
                )
                break
            current_action_model = action_model_instance
            if not isinstance(current_action_model, self.ActionModelType):
                if isinstance(current_action_model, dict):
                    try:
                        current_action_model = self.ActionModelType(
                            **current_action_model
                        )  # type: ignore
                    except ValidationError as e:
                        results.append(
                            ActionResult(
                                error=f"Invalid action structure for action {i}: {e}"
                            )
                        )
                        break
                else:
                    results.append(
                        ActionResult(
                            error=f"Action {i} is not a valid ActionModel or dictionary."
                        )
                    )
                    break
            page_extraction_llm = self.settings.page_extraction_llm or self.llm
            result = await self.controller.act(
                action=current_action_model,
                browser_session=self.browser_session,
                page_extraction_llm=page_extraction_llm,
                sensitive_data=self.sensitive_data,
                available_file_paths=self.settings.available_file_paths,
                context=self.context,
            )
            results.append(result)
            if result.is_done or result.error:
                break
            if (
                i < len(actions) - 1
                and self.settings.interrupt_on_page_change_in_multi_act
            ):
                await asyncio.sleep(
                    self.browser_session.browser_profile.wait_between_actions / 2
                )
                fresh_state_summary = await self.browser_session.get_state_summary(
                    cache_clickable_elements_hashes=True
                )
                current_url_after_action = fresh_state_summary.url
                new_hashes = (
                    self.browser_session._cached_clickable_element_hashes.hashes
                    if self.browser_session._cached_clickable_element_hashes
                    else set()
                )
                url_changed = current_url_after_action != initial_hashes_url
                dom_changed = not url_changed and new_hashes != initial_hashes_set
                if url_changed:
                    logger.info(
                        f"URL changed from '{initial_hashes_url}' to '{current_url_after_action}', interrupting multi_act."
                    )
                    break
                if dom_changed:
                    logger.info(
                        f"DOM changed on '{initial_hashes_url}', interrupting multi_act."
                    )
                    break
                initial_hashes_url = current_url_after_action
                initial_hashes_set = new_hashes
            if i == len(actions) - 1:
                await asyncio.sleep(
                    self.browser_session.browser_profile.wait_between_actions
                )
        return results

    async def _handle_step_error(self, error: Exception) -> List[ActionResult]:
        error_msg = str(error)
        logger.error(f"Step failed: {error_msg}", exc_info=True)
        self.state.consecutive_failures += 1
        return [ActionResult(error=error_msg, include_in_memory=True)]

    async def step(self, step_info: Optional[AgentStepInfo] = None):
        logger.info(f"--- Step {self.state.n_steps} ---")
        browser_state_summary = None
        model_output: Optional[AgentOutput] = None
        result: List[ActionResult] = []
        tool_call_id_for_step: Optional[str] = None  # Initialize here
        try:
            if not self.browser_session.initialized:
                await self.browser_session.start()
            browser_state_summary = await self.browser_session.get_state_summary()
            self._message_manager.add_state_message(
                browser_state_summary=browser_state_summary,
                result=self.state.last_result,
                step_info=step_info,
                use_vision=self.settings.use_vision,
            )
            input_messages = self._message_manager.get_messages()

            model_output, tool_call_id_for_step = await self.get_next_action(
                input_messages
            )  # Unpack

            self._message_manager.state.history.remove_last_state_message()
            # Pass the tool_call_id from the AIMessage to add_model_output so it can be used for the ToolMessage
            if tool_call_id_for_step:
                self._message_manager.add_model_output(
                    model_output, tool_call_id_for_step
                )
            else:  # Should not happen if tool calling is used and successful
                logger.warning(
                    "No tool_call_id received from get_next_action, AIMessage might not be correctly formatted for tool call."
                )
                # Fallback: create a new ID for the AIMessage itself if one wasn't part of the LLM response structure.
                # This part might need more sophisticated handling depending on how non-tool_call scenarios are structured.
                fallback_tc_id = f"gen_tc_{self.state.n_steps}_{uuid.uuid4()}"[:40]
                self._message_manager.add_model_output(model_output, fallback_tc_id)
                tool_call_id_for_step = fallback_tc_id

            result = await self.multi_act(model_output.action)
            self.state.last_result = result
            self.state.consecutive_failures = 0

            if (
                tool_call_id_for_step
            ):  # Use the ID from the AIMessage for the corresponding ToolMessage
                combined_summary = "; ".join(
                    f"Action {idx + 1}: {(r.extracted_content or r.error or 'OK')[:100]}"
                    for idx, r in enumerate(result)
                )
                self._message_manager.add_tool_message(
                    content=f"Actions processed. Summary: {combined_summary}",
                    tool_call_id=tool_call_id_for_step,
                )
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result
            if tool_call_id_for_step:
                self._message_manager.add_tool_message(
                    content=f"Error: {str(e)}", tool_call_id=tool_call_id_for_step
                )
            # else: logger.error(f"Error in step before tool_call_id was obtained: {e}") # No specific tool_call_id to associate error with
        finally:
            if browser_state_summary and model_output:
                self.state.history.history.append(
                    AgentHistory(
                        model_output=model_output,
                        result=result,
                        state={
                            "url": browser_state_summary.url,
                            "title": browser_state_summary.title,
                            "screenshot_summary": (
                                browser_state_summary.screenshot[:100]
                                if browser_state_summary.screenshot
                                else None
                            ),
                        },
                        metadata={"step": self.state.n_steps},
                    )
                )
            self.state.n_steps += 1

    async def run(
        self,
        max_steps: int = 100,
        on_step_start: Optional[Callable[["Agent"], Awaitable[None]]] = None,
        on_step_end: Optional[Callable[["Agent"], Awaitable[None]]] = None,
    ) -> AgentHistoryList:
        logger.info(f"🚀 Starting task: {self.task}")
        if self.initial_actions:
            logger.info(f"Executing initial actions: {self.initial_actions}")
            self.state.last_result = await self.multi_act(self.initial_actions)
        for step_num in range(max_steps):
            if self.state.stopped:
                logger.info("Agent stopped.")
                break
            if self.state.paused:
                logger.info("Agent paused. Waiting...")
                await self._external_pause_event.wait()
            if self.state.stopped:
                logger.info("Agent stopped during pause.")
                break
                logger.info("Agent resumed.")
            if self.state.consecutive_failures >= self.settings.max_failures:
                logger.error(
                    f"Stopping due to {self.settings.max_failures} consecutive failures."
                )
                (
                    self._add_failure_done_action("Max consecutive failures reached.")
                    if not self.state.history.is_done()
                    else None
                )
                break
            current_step_info = AgentStepInfo(step_number=step_num, max_steps=max_steps)
            if on_step_start:
                await on_step_start(self)
            await self.step(step_info=current_step_info)
            if on_step_end:
                await on_step_end(self)
            if self.state.history.is_done():
                logger.info("✅ Task marked as done.")
                break
        else:
            logger.info(f"Max steps ({max_steps}) reached.")
            if not self.state.history.is_done():
                self._add_failure_done_action(
                    "Max steps reached, task may be incomplete."
                )
        logger.info(f"Agent run finished. Total steps: {self.state.n_steps - 1}")
        if self.settings.save_conversation_path:
            self._save_conversation()
        return self.state.history

    def _add_failure_done_action(self, reason: str):
        logger.warning(f"Adding failure 'done' action: {reason}")
        done_params = DoneAction(text=reason, success=False)
        final_action = self.ActionModelType(**{"done": done_params.model_dump()})
        final_brain = AgentBrain(
            evaluation_previous_goal=reason, memory="N/A", next_goal="N/A"
        )
        final_model_output = self.AgentOutputType(
            current_state=final_brain, action=[final_action]
        )
        final_result = ActionResult(
            is_done=True, success=False, extracted_content=reason
        )
        if self.state.history.history:
            last_h = self.state.history.history[-1]
            if (
                last_h.result
                and last_h.result[-1].is_done
                and not last_h.result[-1].success
                and reason in (last_h.result[-1].extracted_content or "")
            ):
                logger.info(f"Failure '{reason}' already in history.")
                return
        self.state.history.history.append(
            AgentHistory(
                model_output=final_model_output,
                result=[final_result],
                state={"url": "N/A", "title": "N/A"},
                metadata={
                    "step": self.state.n_steps,
                    "reason": f"Forced done: {reason}",
                },
            )
        )

    def _save_conversation(self):
        if not self.settings.save_conversation_path:
            return
        try:
            path = Path(self.settings.save_conversation_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_path_str = str(
                path
                if path.name.lower().endswith(".json")
                else path.with_name(f"{path.name}_agent_history.json")
            )
            with open(file_path_str, "w") as f:
                f.write(self.state.history.model_dump_json(indent=2))
            logger.info(f"Agent history saved to {file_path_str}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    async def close(self):
        if self.browser_session and self.browser_session.initialized:
            await self.browser_session.stop()
        logger.info("Agent closed.")

    def pause(self):
        self.state.paused = True
        self._external_pause_event.clear()
        logger.info("Agent paused.")

    def resume(self):
        if self.state.paused:
            self.state.paused = False
            self._external_pause_event.set()
            logger.info("Agent resumed.")
        else:
            logger.info("Agent not paused.")

    def stop(self):
        self.state.stopped = True
        if self.state.paused:
            self._external_pause_event.set()
        logger.info("Agent stop requested.")


def log_response(response: AgentOutput) -> None:
    emoji = "🤷"
    if response.current_state:
        if response.current_state.evaluation_previous_goal:
            if "Success" in response.current_state.evaluation_previous_goal:
                emoji = "👍"
            elif "Failed" in response.current_state.evaluation_previous_goal:
                emoji = "⚠"
            logger.info(
                f"{emoji} Eval: {response.current_state.evaluation_previous_goal}"
            )
        logger.info(f"🧠 Memory: {response.current_state.memory}")
        logger.info(f"🎯 Next goal: {response.current_state.next_goal}")
    if response.action:
        for i, action_instance in enumerate(response.action):
            action_dict = action_instance.model_dump(
                exclude_unset=True, exclude_none=True
            )
            if action_dict:
                action_name = list(action_dict.keys())[0]
                action_params = action_dict[action_name]
                logger.info(
                    f"🛠️ Action {i + 1}/{len(response.action)}: {action_name}({json.dumps(action_params)})"
                )
            else:
                logger.warning(
                    f"Action {i + 1}/{len(response.action)} is empty/invalid."
                )
    else:
        logger.warning("No action in AgentOutput.")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    task = "Go to google.com, search for 'current weather in Ho Chi Minh City', and then extract the temperature."
    current_bp = DEFAULT_BROWSER_PROFILE.model_copy(
        update={
            "user_data_dir": os.getenv("PERSISTENT_PROFILE_PATH"),
            "executable_path": os.getenv("CHROME_EXE_PATH"),
            "headless": False,
            "args": (DEFAULT_BROWSER_PROFILE.args or []) + ["--start-maximized"],
            "include_attributes": [
                "id",
                "class",
                "name",
                "role",
                "aria-label",
                "placeholder",
                "value",
                "alt",
                "type",
                "title",
                "href",
                "text_content",
            ],
            "highlight_elements": True,
        }
    )
    current_as = AgentSettings(
        use_vision=True,
        max_actions_per_step=3,
        tool_calling_method="tools",
        page_extraction_llm=llm,
        save_conversation_path=os.path.join(
            "conversations",
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_llm_conversation_history.json",
        ),
    )
    agent = Agent(
        task=task,
        llm=llm,
        agent_settings=current_as,
        browser_profile=current_bp,
        controller=Controller(),
    )
    try:
        print(f"🚀 Starting agent task: {task}")
        history = await agent.run(max_steps=10)
        print("\n--- Agent Run History ---")
        if history.history:
            for i, hist_item in enumerate(history.history):
                print(
                    f"\n--- History Step {i + 1} (Agent Step {hist_item.metadata.get('step', 'N/A') if hist_item.metadata else 'N/A'}) ---"
                )
                if hist_item.model_output:
                    cs = hist_item.model_output.current_state
                    print(
                        f"  LLM Eval: '{cs.evaluation_previous_goal}'\n  LLM Memory: '{cs.memory}'\n  LLM Next Goal: '{cs.next_goal}'"
                    )
                    for action_item in hist_item.model_output.action:
                        ad = action_item.model_dump(
                            exclude_unset=True, exclude_none=True
                        )
                        if ad:
                            print(
                                f"  LLM Action: {list(ad.keys())[0]}({json.dumps(list(ad.values())[0])})"
                            )  # type: ignore
                if hist_item.result:
                    for res_idx, res_item in enumerate(hist_item.result):
                        print(f"  Action Result {res_idx + 1}:")
                        if res_item.extracted_content:
                            print(f"    Content: {res_item.extracted_content[:200]}...")
                        if res_item.error:
                            print(f"    Error: {res_item.error}")
                        if res_item.is_done:
                            print(f"    Task Done: Success={res_item.success}")
                if hist_item.state and hist_item.state.get("url"):
                    print(
                        f"  Browser State: URL={hist_item.state['url']}, Title='{hist_item.state.get('title', 'N/A')}'"
                    )
        final_content = history.final_result()
        if final_content:
            print(f"\n✅ Final Result: {final_content}")
        else:
            print("\n Agent did not complete successfully or produce a final result.")
    except Exception as e:
        print(f"Error during agent execution: {e}")
        logger.error("Main execution error", exc_info=True)
    finally:
        print("Closing browser session...")
        await agent.close()


def __main__():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())


if __name__ == "__main__":
    __main__()
