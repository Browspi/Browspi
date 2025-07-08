import asyncio
import base64
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
from typing import Tuple
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
    Browser,
    ElementHandle,
    FrameLocator,
    Page,
    Playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
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

filterwarnings("ignore", category=LangChainBetaWarning)
load_dotenv()
logger = logging.getLogger(__name__)


class DOMElementNode(BaseModel):
    xpath: str
    tag_name: str
    attributes: Dict[str, Any] = {}
    highlight_index: Optional[int] = None


class BrowserStateSummary(BaseModel):
    url: str
    title: str
    screenshot: Optional[str] = None
    tabs: List[Dict[str, Any]] = []
    element_tree: Optional[Any] = None
    selector_map: Dict[int, DOMElementNode] = {}
    pixels_above: Optional[int] = None
    pixels_below: Optional[int] = None

    def clickable_elements_to_string(
        self, include_attributes: Optional[List[str]] = None
    ) -> str:
        if not self.selector_map:
            return "No interactive elements found."

        lines = []
        for index, element_info in sorted(self.selector_map.items()):
            line = f"Index {index}: <{element_info.tag_name}>"
            attrs_to_show = include_attributes or [
                "id",
                "name",
                "role",
                "aria-label",
                "text_content",
            ]
            attr_strings = []
            for attr_name in attrs_to_show:
                if (
                    attr_name in element_info.attributes
                    and element_info.attributes[attr_name]
                ):
                    attr_strings.append(
                        f'{attr_name}="{str(element_info.attributes[attr_name])[:30]}"'
                    )
            if attr_strings:
                line += f" [{', '.join(attr_strings)}]"

            text_content = element_info.attributes.get(
                "text_content"
            ) or element_info.attributes.get("aria-label")
            if text_content:
                line += f" (Text: '{str(text_content)[:50]}...')"

            lines.append(line)
        return "\n".join(lines)


class BrowserConfig(BaseModel):
    user_data_dir: Optional[str] = None
    executable_path: Optional[str] = None
    headless: bool = True
    args: List[str] = []
    include_attributes: List[str] = Field(
        default_factory=lambda: [
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
        ]
    )
    highlight_elements: bool = False
    wait_between_actions: float = 0.5
    cookies_file: Optional[str] = None
    storage_state: Optional[str] = None
    viewport: Optional[Dict[str, int]] = None


DEFAULT_BROWSER_PROFILE = BrowserConfig()


class WebNavigator:
    def __init__(self, browser_profile: BrowserConfig):
        self.browser_profile = browser_profile
        self.playwright_context: Optional[Any] = None
        self.browser: Optional[Browser] = None
        self.playwright: Optional[Playwright] = None
        self.agent_current_page: Optional[Page] = None
        self.initialized: bool = False
        self._cached_browser_state_summary: Optional[BrowserStateSummary] = None
        self._cached_clickable_element_hashes: Optional[Any] = None

    async def start(self):
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()
        p = self.playwright

        if self.browser_profile.user_data_dir:
            logger.info(
                f"Attempting to launch persistent context with user_data_dir: {self.browser_profile.user_data_dir}"
            )
            try:
                self.playwright_context = await p.chromium.launch_persistent_context(
                    user_data_dir=self.browser_profile.user_data_dir,
                    headless=self.browser_profile.headless,
                    executable_path=self.browser_profile.executable_path,
                    args=self.browser_profile.args,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    viewport=self.browser_profile.viewport,
                )
                if self.playwright_context.pages:
                    self.agent_current_page = self.playwright_context.pages[0]
                else:
                    self.agent_current_page = await self.playwright_context.new_page()
                logger.info(
                    f"Launched persistent context with profile: {self.browser_profile.user_data_dir}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to launch persistent context with user_data_dir '{self.browser_profile.user_data_dir}': {e}",
                    exc_info=True,
                )
                logger.info("Falling back to non-persistent context launch.")
                await self._launch_non_persistent_context(p)
        else:
            logger.info(
                "Launching non-persistent context (no user_data_dir specified in profile)."
            )
            await self._launch_non_persistent_context(p)

        self.initialized = True
        logger.info("Browser session started.")

    async def _launch_non_persistent_context(self, playwright_instance):
        """Helper method to launch a non-persistent browser context."""
        self.browser = await playwright_instance.chromium.launch(
            headless=self.browser_profile.headless,
            executable_path=self.browser_profile.executable_path,
            args=self.browser_profile.args,
        )
        self.playwright_context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport=self.browser_profile.viewport,
        )
        if self.browser_profile.storage_state and os.path.exists(
            self.browser_profile.storage_state
        ):
            logger.info(
                f"Loading storage state for new context from: {self.browser_profile.storage_state}"
            )
            try:
                with open(self.browser_profile.storage_state, "r") as f:
                    storage_state_dict = json.load(f)
                await self.playwright_context.add_cookies(
                    storage_state_dict.get("cookies", [])
                )
            except Exception as e:
                logger.error(
                    f"Failed to load storage state from '{self.browser_profile.storage_state}': {e}",
                    exc_info=True,
                )

        self.agent_current_page = await self.playwright_context.new_page()
        logger.info("Launched non-persistent context.")

    async def stop(self):
        if self.playwright_context:
            await self.playwright_context.close()
            self.playwright_context = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        logger.info("Browser session stopped.")

    async def get_current_page(self) -> Page:
        if not self.agent_current_page or self.agent_current_page.is_closed():
            if self.playwright_context and self.playwright_context.pages:
                self.agent_current_page = self.playwright_context.pages[0]
            elif self.playwright_context:
                self.agent_current_page = await self.playwright_context.new_page()
            else:
                raise PlaywrightError("Browser context not available to get a page.")
        if not self.agent_current_page:
            raise PlaywrightError("Failed to get or create a current page.")
        return self.agent_current_page

    async def navigate(self, url: str):
        page = await self.get_current_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

    async def remove_highlights(self):
        page = await self.get_current_page()
        try:
            await page.evaluate(
                """
                () => {
                    const container = document.getElementById('playwright-highlight-container');
                    if (container) {
                        container.remove();
                    }
                }
                """
            )
        except Exception as e:
            logger.debug(f"Could not remove highlights: {e}")

    async def get_state_summary(
        self, cache_clickable_elements_hashes: bool = True
    ) -> BrowserStateSummary:
        page = await self.get_current_page()
        title = await page.title()
        url = page.url

        if self.browser_profile.highlight_elements:
            await self.remove_highlights()

        selector_map_mock: Dict[int, Any] = {}
        try:
            from browspi.services.dom.service import DomService

            dom_service = DomService(page)
            dom_state = await dom_service.get_clickable_elements(
                highlight_elements=True,
            )
            selector_map_mock = dom_state.selector_map
        except Exception as e:
            logger.error(f"Error extracting elements for mock selector_map: {e}")

        screenshot_b64 = None
        try:
            screenshot_bytes = await page.screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"Could not take screenshot: {e}")

        selector_map_for_pydantic = {k: v.__dict__ for k, v in selector_map_mock.items()}

        summary = BrowserStateSummary(
            url=url,
            title=title,
            selector_map=selector_map_for_pydantic,
            screenshot=screenshot_b64,
        )
        self._cached_browser_state_summary = summary

        if cache_clickable_elements_hashes:
            current_hashes = {info.get('xpath') for info in selector_map_for_pydantic.values()}
            if self._cached_clickable_element_hashes:
                pass
            self._cached_clickable_element_hashes = {
                "url": url,
                "hashes": current_hashes,
            }

        return summary

    async def _get_xpath(self, element: ElementHandle) -> str:
        return await element.evaluate(
            """
            el => {
                if (!el || el.nodeType !== 1) return '';
                const paths = [];
                for (; el && el.nodeType === 1; el = el.parentNode) {
                    let index = 0;
                    for (let sibling = el.previousSibling; sibling; sibling = sibling.previousSibling) {
                        if (sibling.nodeType === 1 && sibling.nodeName === el.nodeName) {
                            index++;
                        }
                    }
                    const tagName = el.nodeName.toLowerCase();
                    const pathIndex = (index ? `[${index + 1}]` : '');
                    paths.splice(0, 0, tagName + pathIndex);
                }
                return paths.length ? '/' + paths.join('/') : '';
            }
        """
        )

    async def close_tab(self, page_id: int):
        if self.playwright_context and 0 <= page_id < len(
            self.playwright_context.pages
        ):
            page_to_close = self.playwright_context.pages[page_id]
            if page_to_close == self.agent_current_page:
                self.agent_current_page = None
            await page_to_close.close()
            logger.info(f"Closed tab with index: {page_id}")
        else:
            logger.warning(
                f"Tab index {page_id} out of range or context not available."
            )

    SelectorMap = dict[int, DOMElementNode]

    async def get_selector_map(self) -> SelectorMap:
        if self._cached_browser_state_summary is None:
            return {}
        return self._cached_browser_state_summary.selector_map

    async def find_file_upload_element_by_index(
        self, index: int
    ) -> DOMElementNode | None:
        """
        Find a file upload element related to the element at the given index:
        - Check if the element itself is a file input
        - Check if it's a label pointing to a file input
        - Recursively search children for file inputs
        - Check siblings for file inputs

        Args:
            index: The index of the candidate element (could be a file input, label, or parent element)

        Returns:
            The DOM element for the file input if found, None otherwise
        """
        try:
            selector_map = await self.get_selector_map()
            if index not in selector_map:
                return None

            candidate_element = selector_map[index]

            def is_file_input(node: DOMElementNode) -> bool:
                return (
                    isinstance(node, DOMElementNode)
                    and node.tag_name == "input"
                    and node.attributes.get("type") == "file"
                )

            def find_element_by_id(
                node: DOMElementNode, element_id: str
            ) -> DOMElementNode | None:
                if isinstance(node, DOMElementNode):
                    if node.attributes.get("id") == element_id:
                        return node
                    for child in node.children:
                        result = find_element_by_id(child, element_id)
                        if result:
                            return result
                return None

            def get_root(node: DOMElementNode) -> DOMElementNode:
                root = node
                while root.parent:
                    root = root.parent
                return root

            def find_file_input_recursive(
                node: DOMElementNode, max_depth: int = 3, current_depth: int = 0
            ) -> DOMElementNode | None:
                if current_depth > max_depth or not isinstance(node, DOMElementNode):
                    return None

                if is_file_input(node):
                    return node

                if node.children and current_depth < max_depth:
                    for child in node.children:
                        if isinstance(child, DOMElementNode):
                            result = find_file_input_recursive(
                                child, max_depth, current_depth + 1
                            )
                            if result:
                                return result
                return None

            if is_file_input(candidate_element):
                return candidate_element

            if (
                candidate_element.tag_name == "label"
                and candidate_element.attributes.get("for")
            ):
                input_id = candidate_element.attributes.get("for")
                root_element = get_root(candidate_element)

                target_input = find_element_by_id(root_element, input_id)
                if target_input and is_file_input(target_input):
                    return target_input

            child_result = find_file_input_recursive(candidate_element)
            if child_result:
                return child_result

            if candidate_element.parent:
                for sibling in candidate_element.parent.children:
                    if sibling is not candidate_element and isinstance(
                        sibling, DOMElementNode
                    ):
                        if is_file_input(sibling):
                            return sibling
            return None

        except Exception as e:
            logger.debug(f"Error in find_file_upload_element_by_index: {e}")
            return None

    async def get_locate_element(self, element: DOMElementNode) -> ElementHandle | None:
        page = await self.get_current_page()
        current_frame = page

        parents: list[DOMElementNode] = []
        current = element
        while current.parent is not None:
            parent = current.parent
            parents.append(parent)
            current = parent

        parents.reverse()

        iframes = [item for item in parents if item.tag_name == "iframe"]
        for parent in iframes:
            css_selector = self._enhanced_css_selector_for_element(
                parent,
                include_dynamic_attributes=self.browser_profile.include_dynamic_attributes,
            )
            current_frame = current_frame.frame_locator(css_selector)

        css_selector = self._enhanced_css_selector_for_element(
            element,
            include_dynamic_attributes=self.browser_profile.include_dynamic_attributes,
        )

        try:
            if isinstance(current_frame, FrameLocator):
                element_handle = await current_frame.locator(
                    css_selector
                ).element_handle()
                return element_handle
            else:
                element_handle = await current_frame.query_selector(css_selector)
                if element_handle:
                    is_visible = await self._is_visible(element_handle)
                    if is_visible:
                        await element_handle.scroll_into_view_if_needed()
                    return element_handle
                return None
        except Exception as e:
            logger.error(f"❌  Failed to locate element: {str(e)}")
            return None


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
        logging.getLogger().handlers = []

    root = logging.getLogger()

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
    main_logger.handlers = []
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


class StepResult(BaseModel):
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
                return value.index
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
                return
        logger.warning(f"Could not set index on action: {self}")


class GoToUrlAction(BaseModel):
    url: str


class SearchGoogleAction(BaseModel):
    query: str


class ClickElementAction(BaseModel):
    index: int


class InputTextAction(BaseModel):
    """
    Action to input text into a specific element on the page.
    """
    text: str
    selector: Optional[str] = None 
    index: Optional[int] = None   

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
            )
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
                continue
            if page and action_info.page_filter and not action_info.page_filter(page):
                continue
            fields[name] = (
                Optional[action_info.param_model],
                Field(default=None, description=action_info.description),
            )
        return create_model("DynamicActionModel", __base__=ActionModel, **fields)


class ActionManager(Generic[Context]):
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
                success: bool

            self.registry.actions["done"] = RegisteredAction(
                name="done",
                description="Completes the task with custom output",
                function=self._custom_done_action_func,
                param_model=CustomDoneAction,
            )

    async def _custom_done_action_func(self, params: BaseModel):
        return StepResult(
            is_done=True,
            success=params.success,
            extracted_content=params.data.model_dump_json(),
        )

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
                    if name
                    not in [
                        "browser_session",
                        "page_extraction_llm",
                        "context",
                        "self",
                    ]
                }
                actual_param_model = create_model(f"{func.__name__}Params", **fields)
            self.registry.actions[func.__name__] = RegisteredAction(
                name=func.__name__,
                description=description,
                function=func,
                param_model=actual_param_model,
                **kwargs,
            )
            return func

        return decorator

    def _register_default_actions(self):
        @self.action("Navigate to a URL", param_model=GoToUrlAction)
        async def go_to_url(params: GoToUrlAction, browser_session: WebNavigator):
            await browser_session.navigate(params.url)
            return StepResult(
                extracted_content=f"Navigated to {params.url}", include_in_memory=True
            )

        @self.action("Search Google", param_model=SearchGoogleAction)
        async def search_google(
            params: SearchGoogleAction, browser_session: WebNavigator
        ):
            await browser_session.navigate(
                f"https://www.google.com/search?q={params.query.replace(' ', '+')}"
            )
            return StepResult(
                extracted_content=f"Searched Google for: {params.query}",
                include_in_memory=True,
            )

        @self.action("Click an element by its index", param_model=ClickElementAction)
        async def click_element_by_index(
            params: ClickElementAction, browser_session: WebNavigator
        ):
            logger.info(f"Attempting to click element with index: {params.index}")
            if not browser_session._cached_browser_state_summary:
                return StepResult(
                    error=f"Browser state not yet summarized. Cannot find element {params.index}"
                )

            state = browser_session._cached_browser_state_summary

            if params.index not in state.selector_map:
                return StepResult(
                    error=f"Element with index {params.index} not found in selector_map."
                )

            element_info = state.selector_map[params.index]
            page = await browser_session.get_current_page()

            base_locator = page.locator(f"xpath={element_info.xpath}")

            final_element_handle: Optional[ElementHandle] = None

            candidate_handles_to_dispose: List[ElementHandle] = []

            try:
                matching_elements = await base_locator.element_handles()
                candidate_handles_to_dispose.extend(matching_elements)

                if not matching_elements:
                    return StepResult(
                        error=f"No elements found for XPath: {element_info.xpath} (index {params.index})."
                    )

                if len(matching_elements) == 1:
                    final_element_handle = matching_elements[0]
                    logger.info(
                        f"XPath {element_info.xpath} for index {params.index} resolved to a unique element."
                    )
                else:
                    logger.warning(
                        f"XPath {element_info.xpath} for index {params.index} resolved to {len(matching_elements)} elements. Attempting to disambiguate."
                    )

                    attr_id_val = element_info.attributes.get("id")
                    attr_name_val = element_info.attributes.get("name")
                    attr_value_val = element_info.attributes.get("value")
                    attr_aria_label_val = element_info.attributes.get("aria-label")
                    attr_text_content_val = (
                        element_info.attributes.get("text_content") or ""
                    ).strip()

                    found_specific_match_idx = -1

                    for i, el_handle_candidate in enumerate(matching_elements):
                        try:
                            candidate_id = await el_handle_candidate.get_attribute("id")
                            candidate_name = await el_handle_candidate.get_attribute(
                                "name"
                            )
                            candidate_value = await el_handle_candidate.get_attribute(
                                "value"
                            )
                            candidate_aria_label = (
                                await el_handle_candidate.get_attribute("aria-label")
                            )
                            candidate_text = (
                                await el_handle_candidate.text_content() or ""
                            ).strip()

                            matches_criteria = True
                            if attr_id_val is not None and candidate_id != attr_id_val:
                                matches_criteria = False
                            elif (
                                attr_name_val is not None
                                and candidate_name != attr_name_val
                            ):
                                matches_criteria = False
                            elif (
                                attr_value_val is not None
                                and candidate_value != attr_value_val
                            ):
                                matches_criteria = False
                            elif (
                                attr_aria_label_val is not None
                                and candidate_aria_label != attr_aria_label_val
                            ):
                                matches_criteria = False

                            elif (
                                attr_text_content_val
                                and candidate_text != attr_text_content_val
                            ):
                                matches_criteria = False

                            if matches_criteria:
                                final_element_handle = el_handle_candidate
                                found_specific_match_idx = i
                                logger.info(
                                    f"Disambiguated to element #{i + 1} of {len(matching_elements)} for index {params.index} using stored attributes."
                                )
                                break
                        except Exception as eval_err:
                            logger.warning(
                                f"Error evaluating attributes for candidate element #{i + 1}: {eval_err}"
                            )
                            continue

                    if not final_element_handle:
                        logger.warning(
                            f"Could not uniquely disambiguate element for index {params.index}. Falling back to the first element."
                        )
                        if matching_elements:
                            final_element_handle = matching_elements[0]
                            found_specific_match_idx = 0

                    temp_dispose_list = []
                    if final_element_handle:
                        for i, h in enumerate(candidate_handles_to_dispose):
                            if (
                                h == final_element_handle
                                and i == found_specific_match_idx
                            ):
                                continue
                            temp_dispose_list.append(h)
                    else:
                        temp_dispose_list.extend(candidate_handles_to_dispose)

                    for h_to_dispose in temp_dispose_list:
                        try:
                            await h_to_dispose.dispose()
                        except Exception:
                            pass

                    if (
                        final_element_handle
                        and final_element_handle not in temp_dispose_list
                    ):
                        candidate_handles_to_dispose = [final_element_handle]
                    else:
                        candidate_handles_to_dispose = []

                if not final_element_handle:
                    return StepResult(
                        error=f"Could not obtain a specific element handle for index {params.index} (xpath: {element_info.xpath})."
                    )

                if not await final_element_handle.is_visible():
                    logger.warning(
                        f"Element {params.index} (xpath: {element_info.xpath}) is not visible. Attempting click anyway."
                    )

                try:
                    logger.debug(
                        f"Attempting Playwright click for refined element {params.index} (xpath: {element_info.xpath})"
                    )
                    await final_element_handle.click(timeout=3000)
                    logger.info(
                        f"Successfully clicked element {params.index} using Playwright click."
                    )
                    return StepResult(
                        extracted_content=f"Clicked element at index {params.index} ({element_info.tag_name})",
                        include_in_memory=True,
                    )
                except PlaywrightTimeoutError:
                    logger.warning(
                        f"Playwright click timed out for element {params.index}. Trying JS click."
                    )
                except PlaywrightError as pe_click:
                    logger.warning(
                        f"Playwright click failed for element {params.index}: {pe_click}. Trying JS click."
                    )

                try:
                    logger.debug(
                        f"Attempting JavaScript click for element {params.index} (xpath: {element_info.xpath})"
                    )
                    await page.evaluate("(el) => el.click()", final_element_handle)
                    logger.info(
                        f"Successfully clicked element {params.index} using JavaScript click."
                    )
                    return StepResult(
                        extracted_content=f"Clicked element at index {params.index} ({element_info.tag_name}) via JS",
                        include_in_memory=True,
                    )
                except Exception as js_e:
                    logger.error(
                        f"JavaScript click failed for element {params.index}: {js_e}"
                    )
                    return StepResult(
                        error=f"Failed to click element at index {params.index} (xpath: {element_info.xpath}) using both methods: JS click error: {js_e}"
                    )
            except PlaywrightTimeoutError:
                logger.error(
                    f"Timeout occurred in click_element_by_index for XPath: {element_info.xpath} (index {params.index})."
                )
                return StepResult(
                    error=f"Timeout resolving locator or operation for XPath: {element_info.xpath} (index {params.index})."
                )
            except Exception as e:
                logger.error(
                    f"General error in click_element_by_index for {params.index} (xpath: {element_info.xpath}): {type(e).__name__} - {e}",
                    exc_info=True,
                )
                return StepResult(
                    error=f"Failed to click element at index {params.index} (xpath: {element_info.xpath}): {type(e).__name__} - {e}"
                )
            finally:
                all_handles_to_check_dispose = []
                if final_element_handle:
                    all_handles_to_check_dispose.append(final_element_handle)

                all_handles_to_check_dispose.extend(candidate_handles_to_dispose)

                unique_handles_to_dispose = []
                seen_handles = set()
                for h in all_handles_to_check_dispose:
                    if h not in seen_handles:
                        unique_handles_to_dispose.append(h)
                        seen_handles.add(h)

                for h_to_dispose in unique_handles_to_dispose:
                    try:
                        await h_to_dispose.dispose()
                    except Exception:
                        pass

        @self.action("Input text into an element", param_model=InputTextAction)
        async def input_text(params: InputTextAction, browser_session: WebNavigator):
            """
            Inputs text into a specified element, handling standard inputs and contenteditable divs.

            This function tries multiple methods to input text:
            1. Tries to `fill` the element, which works for standard <input>, <textarea>.
            2. If fill fails, it tries to `type` into the element.
            3. If both fail, it uses JavaScript to set the element's innerHTML, which is
            effective for contenteditable divs like in Gmail's reply box.
            """
            selector = getattr(params, 'selector', None)
            index = getattr(params, 'index', None)
            
            logger.info(
                f"Attempting to input text '{params.text}' into element with selector: '{selector}' or index: {index}"
            )
            page = await browser_session.get_current_page()
            element_handle = None

            # Locate the element using selector or index
            try:
                if selector:
                    element_handle = await page.locator(selector).first.element_handle()
                elif index is not None:
                    state = browser_session._cached_browser_state_summary
                    if not state or index not in state.selector_map:
                        return StepResult(error=f"Element with index {index} not found.")
                    element_info = state.selector_map[index]
                    element_handle = await page.locator(f"xpath={element_info.xpath}").first.element_handle()
            except Exception as e:
                logger.error(f"Could not find element: {e}")
                return StepResult(error=f"Element not found with selector '{selector}' or index '{index}'.")

            if not element_handle:
                return StepResult(error="No element found to input text into.")

            # --- Start of Input Logic ---
            try:
                # Method 1: Try to fill (best for standard inputs)
                await element_handle.fill(params.text, timeout=3000)
                logger.info("Successfully input text using 'fill'.")
                return StepResult(extracted_content="Inputted text successfully.", include_in_memory=True)
            except Exception as e:
                logger.warning(f"Playwright 'fill' failed: {e}. Trying 'type'.")
                try:
                    # Method 2: Try to type (good for some dynamic fields)
                    await element_handle.focus()
                    await element_handle.type(params.text, delay=50) # Add a small delay for stability
                    logger.info("Successfully input text using 'type'.")
                    return StepResult(extracted_content="Inputted text successfully.", include_in_memory=True)
                except Exception as e2:
                    logger.warning(f"Playwright 'type' failed: {e2}. Trying JavaScript execution.")
                    try:
                        # Method 3: Use JavaScript (robust for contenteditable divs)
                        await element_handle.evaluate("element => element.innerHTML = arguments[0]", params.text)
                        logger.info("Successfully input text using JavaScript execution.")
                        return StepResult(extracted_content="Inputted text successfully.", include_in_memory=True)
                    except Exception as e3:
                        logger.error(f"All input methods failed: {e3}")
                        return StepResult(error=f"Failed to input text into the element using all available methods: {e3}")

        @self.action("Scroll down the page", param_model=ScrollAction)
        async def scroll_down(params: ScrollAction, browser_session: WebNavigator):
            page = await browser_session.get_current_page()
            amount = (
                params.amount if params.amount is not None else "window.innerHeight"
            )
            await page.evaluate(f"window.scrollBy(0, {amount})")
            return StepResult(
                extracted_content=f"Scrolled down by {params.amount or 'one page'}",
                include_in_memory=True,
            )

        @self.action("Scroll up the page", param_model=ScrollAction)
        async def scroll_up(params: ScrollAction, browser_session: WebNavigator):
            page = await browser_session.get_current_page()
            amount = (
                params.amount if params.amount is not None else "window.innerHeight"
            )
            await page.evaluate(f"window.scrollBy(0, -{amount})")
            return StepResult(
                extracted_content=f"Scrolled up by {params.amount or 'one page'}",
                include_in_memory=True,
            )

        @self.action("Mark task as done", param_model=DoneAction)
        async def done(params: DoneAction):
            return StepResult(
                is_done=True, success=params.success, extracted_content=params.text
            )

        @self.action("Open URL in a new tab", param_model=OpenTabAction)
        async def open_tab(params: OpenTabAction, browser_session: WebNavigator):
            if not browser_session.playwright_context:
                await browser_session.start()
            if not browser_session.playwright_context:
                return StepResult(error="Browser context could not be initialized.")
            new_page = await browser_session.playwright_context.new_page()
            await new_page.goto(params.url, wait_until="domcontentloaded")
            browser_session.agent_current_page = new_page
            return StepResult(
                extracted_content=f"Opened new tab with URL: {params.url}",
                include_in_memory=True,
            )

        @self.action("Close an existing tab by its ID", param_model=CloseTabAction)
        async def close_tab_action(
            params: CloseTabAction, browser_session: WebNavigator
        ):
            await browser_session.close_tab(params.page_id)
            return StepResult(
                extracted_content=f"Closed tab with ID: {params.page_id}",
                include_in_memory=True,
            )

        @self.action("Switch to a specific tab by its ID", param_model=SwitchTabAction)
        async def switch_tab(params: SwitchTabAction, browser_session: WebNavigator):
            if not browser_session.playwright_context:
                return StepResult(error="Browser context not available.")
            pages = browser_session.playwright_context.pages
            if 0 <= params.page_id < len(pages):
                browser_session.agent_current_page = pages[params.page_id]
                await browser_session.agent_current_page.bring_to_front()
                return StepResult(
                    extracted_content=f"Switched to tab ID: {params.page_id}",
                    include_in_memory=True,
                )
            return StepResult(error=f"Tab ID {params.page_id} not found.")

        @self.action(
            "Extract content from the current page based on a goal",
            param_model=ExtractPageContentAction,
        )
        async def extract_content(
            params: ExtractPageContentAction,
            browser_session: WebNavigator,
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
                return StepResult(
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
                sub_llm_prompt = f'Please review the following TEXT_TO_PROCESS to fulfill the goal: "{simplified_extraction_goal}". \nExtract the specific information requested. If the goal is to find articles, list their titles and direct URLs. \nIf no specific information or articles matching the goal are found in the text, clearly state that. \n\nTEXT_TO_PROCESS:\n"""{text_to_process}"""'
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
                        or "i cannot directly access or browse urls"
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

            return StepResult(
                extracted_content=extracted_data_summary, include_in_memory=True
            )

        @self.action(
            "Send special keys or keyboard shortcuts", param_model=SendKeysAction
        )
        async def send_keys(params: SendKeysAction, browser_session: WebNavigator):
            page = await browser_session.get_current_page()
            await page.keyboard.press(params.keys)
            return StepResult(
                extracted_content=f"Sent keys: {params.keys}", include_in_memory=True
            )

        @self.action("Wait for a specified number of seconds", param_model=WaitAction)
        async def wait(params: WaitAction):
            await asyncio.sleep(params.seconds)
            return StepResult(
                extracted_content=f"Waited for {params.seconds} seconds",
                include_in_memory=True,
            )

    async def act(
        self,
        action: ActionModel,
        browser_session: WebNavigator,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[List[str]] = None,
        context: Optional[Context] = None,
    ):
        dumped_action = action.model_dump(exclude_unset=True)
        if not dumped_action:
            return StepResult(error="No action in model.")
        action_name = list(dumped_action.keys())[0]
        params_obj = dumped_action[action_name]
        if action_name not in self.registry.actions:
            return StepResult(error=f"Action '{action_name}' not found.")
        registered_action = self.registry.actions[action_name]
        action_kwargs: Dict[str, Any] = {}
        sig = inspect.signature(registered_action.function)

        if "browser_session" in sig.parameters:
            action_kwargs["browser_session"] = browser_session
        if "page_extraction_llm" in sig.parameters:
            action_kwargs["page_extraction_llm"] = page_extraction_llm
        if "sensitive_data" in sig.parameters:
            action_kwargs["sensitive_data"] = sensitive_data
        if "available_file_paths" in sig.parameters:
            action_kwargs["available_file_paths"] = available_file_paths
        if "context" in sig.parameters:
            action_kwargs["context"] = context

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

            param_names_in_model = (
                list(validated_params.model_fields.keys())
                if hasattr(validated_params, "model_fields")
                else []
            )

            if (
                len(param_names_in_model) == 1
                and param_names_in_model[0] in sig.parameters
            ):
                action_kwargs[param_names_in_model[0]] = getattr(
                    validated_params, param_names_in_model[0]
                )
                result = await registered_action.function(**action_kwargs)

            elif (
                all(p_name in sig.parameters for p_name in param_names_in_model)
                and param_names_in_model
            ):
                action_kwargs.update(validated_params.model_dump(exclude_none=True))
                result = await registered_action.function(**action_kwargs)

            elif any(p_name == "params" for p_name in sig.parameters) or (
                len(sig.parameters) - len(action_kwargs) == 1
                and list(sig.parameters.keys())[0] != "self"
            ):
                first_arg_name = next(
                    iter(
                        p
                        for p in sig.parameters
                        if p not in action_kwargs and p != "self"
                    ),
                    None,
                )
                if first_arg_name:
                    result = await registered_action.function(
                        validated_params, **action_kwargs
                    )
                else:
                    result = await registered_action.function(**action_kwargs)

            else:
                result = await registered_action.function(**action_kwargs)

            if isinstance(result, StepResult):
                return result
            return (
                StepResult(extracted_content=str(result))
                if isinstance(result, str)
                else StepResult()
            )
        except PlaywrightTimeoutError as pte:
            logger.error(f"Timeout executing action '{action_name}': {pte}")
            return StepResult(error=f"Action '{action_name}' timed out: {pte}")
        except PlaywrightError as e:
            logger.error(f"Playwright error during action {action_name}: {e}")
            return StepResult(error=f"Browser error during '{action_name}': {e}")
        except ValidationError as e:
            logger.error(
                f"Validation error for action parameters of '{action_name}': {e}"
            )
            return StepResult(error=f"Invalid parameters for '{action_name}': {e}")
        except Exception as e:
            logger.error(f"Error in action {action_name}: {e}", exc_info=True)
            return StepResult(error=f"Unexpected error in '{action_name}': {e}")


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
            logger.debug(
                f"Removed last state message. Current tokens: {self.current_tokens}"
            )


class MessageManagerState(BaseModel):
    history: MessageHistory = Field(default_factory=MessageHistory)
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
        self.system_prompt_message = system_message

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
            from copy import deepcopy

            message_to_add = deepcopy(message)

            if isinstance(message_to_add.content, str):
                temp_content = message_to_add.content
                for placeholder, real_value in self.settings.sensitive_data.items():
                    if real_value:
                        temp_content = temp_content.replace(
                            real_value, f"<secret>{placeholder}</secret>"
                        )
                message_to_add.content = temp_content
            elif isinstance(message_to_add.content, list):
                new_content_list = []
                for item in message_to_add.content:
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
                message_to_add.content = new_content_list
        else:
            message_to_add = message

        self.state.history.add_message(
            message_to_add,
            MessageMetadata(
                tokens=self._count_tokens(message_to_add), message_type=message_type
            ),
            position,
        )
        self._ensure_token_limit()

    def _init_messages(self):
        self._add_message_with_tokens(self.system_prompt_message, message_type="init")
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
                    content=f"Sensitive data placeholders available: {list(self.settings.sensitive_data.keys())}. When you encounter these values in the page content, refer to them using <secret>placeholder_name</secret> format in your reasoning and output. Do not include the actual sensitive values in your responses."
                ),
                message_type="init",
            )

        example_current_state = {
            "evaluation_previous_goal": "Briefly evaluate previous action (Success/Failed/Unknown, and why).",
            "memory": "Summarize what has been done and what critical information was found so far relevant to the main task.",
            "next_goal": "Clearly state the immediate next sub-goal or question to address.",
        }
        example_action_list = [
            {"one_action_name": {"parameter_name": "value"}},
        ]
        example_agent_output_args = {
            "current_state": example_current_state,
            "action": example_action_list,
        }
        self._add_message_with_tokens(
            HumanMessage(
                content=f"You must respond with a single tool call to 'NextAction'. The arguments to this tool call must be a JSON object matching this structure:\n```json\n{json.dumps(example_agent_output_args, indent=2)}\n```\nAlways provide the `current_state` and at least one `action` in the `action` list."
            ),
            message_type="init",
        )

        example_tool_call_id = f"call_ex_{str(uuid.uuid4())[:8]}"
        example_ai_msg = AIMessage(
            content="Okay, I will perform the action based on the provided state.",
            tool_calls=[
                {
                    "id": example_tool_call_id,
                    "name": "NextAction",
                    "args": example_agent_output_args,
                }
            ],
        )
        self._add_message_with_tokens(example_ai_msg, message_type="init")

        self.add_tool_message(
            content="Example action processed. The page updated, and new elements are now visible.",
            tool_call_id=example_tool_call_id,
            message_type="init",
        )

        self._add_message_with_tokens(
            HumanMessage(content="[Task-specific conversation history begins now.]"),
            message_type="init",
        )

        if self.settings.available_file_paths:
            self._add_message_with_tokens(
                HumanMessage(
                    content=f"You have access to these local files if needed by an action: {self.settings.available_file_paths}"
                ),
                message_type="init",
            )

    def add_new_task(self, new_task: str):
        self.task = new_task
        self._add_message_with_tokens(
            HumanMessage(
                content=f'The task has been updated. New task: """{new_task}""". Please consider the previous conversation history as relevant context for this new task.'
            )
        )

    def add_state_message(
        self,
        browser_state_summary: BrowserStateSummary,
        result: list[StepResult] | None = None,
        step_info: Optional[AgentStepInfo] = None,
        use_vision=True,
    ):
        result_summary_parts = []
        if result:
            for i, r_item in enumerate(result):
                if r_item.include_in_memory:
                    if r_item.extracted_content:
                        result_summary_parts.append(
                            f"Result of prior action {i + 1}: {str(r_item.extracted_content)[:200]}"
                        )
                    if r_item.error:
                        result_summary_parts.append(
                            f"Error from prior action {i + 1}: ...{r_item.error.splitlines()[-1][:200]}"
                        )

        result_prefix = ""
        if result_summary_parts:
            result_prefix = (
                "[Prior Action Results]\n" + "\n".join(result_summary_parts) + "\n\n"
            )

        assert browser_state_summary is not None, (
            "BrowserStateSummary cannot be None when adding state message"
        )

        agent_message_prompt = AgentMessagePrompt(
            browser_state_summary=browser_state_summary,
            result=None,
            include_attributes=self.settings.include_attributes,
            step_info=step_info,
        )
        state_desc_text = agent_message_prompt.get_user_message_text_part()

        full_state_description = result_prefix + state_desc_text

        if browser_state_summary.screenshot and use_vision:
            content_list = [
                {"type": "text", "text": full_state_description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{browser_state_summary.screenshot}"
                    },
                },
            ]
            self._add_message_with_tokens(HumanMessage(content=content_list))
        else:
            self._add_message_with_tokens(HumanMessage(content=full_state_description))

    def add_model_output(self, model_output: "NextAction", tool_call_id: str):
        tool_calls = [
            {
                "id": tool_call_id,
                "name": "NextAction",
                "args": model_output.model_dump(exclude_unset=True, exclude_none=True),
            }
        ]
        content = "Okay, proceeding with the determined action(s)."

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        self._add_message_with_tokens(ai_message)

    def add_tool_message(
        self, content: str, tool_call_id: str, message_type: Optional[str] = None
    ):
        tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
        self._add_message_with_tokens(tool_message, message_type=message_type)

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
            removed_message = False
            for i in range(len(self.state.history.messages)):
                if self.state.history.messages[i].metadata.message_type != "init":
                    current_msg_container = self.state.history.messages[i]
                    if (
                        isinstance(current_msg_container.message, AIMessage)
                        and current_msg_container.message.tool_calls
                    ):
                        if i + 1 >= len(self.state.history.messages) or not (
                            isinstance(
                                self.state.history.messages[i + 1].message, ToolMessage
                            )
                            and self.state.history.messages[i + 1].message.tool_call_id
                            == current_msg_container.message.tool_calls[0].get("id")
                        ):
                            logger.debug(
                                f"Skipping removal of AIMessage at index {i} as its ToolMessage is missing or not next."
                            )
                            continue

                    removed_msg_container = self.state.history.messages.pop(i)
                    self.state.history.current_tokens -= (
                        removed_msg_container.metadata.tokens
                    )
                    logger.info(
                        f"Token limit exceeded. Removed message (type: {removed_msg_container.metadata.message_type}, content: '{str(removed_msg_container.message.content)[:50]}...') New token count: {self.state.history.current_tokens}"
                    )
                    removed_message = True
                    break

            if not removed_message:
                logger.warning(
                    f"Could not reduce token count further. Remaining messages: {len(self.state.history.messages)}, Current tokens: {self.state.history.current_tokens}. This might lead to issues."
                )
                break

    def cut_messages(self):
        pass


PROMPT_TEMPLATE = """
You are a proficient AI agent designed to interact with web pages based on user tasks.
Your goal is to understand the current state of a web page and decide the best next action(s) to achieve the user's objective.

You will be provided with:
1.  The current URL and title of the page.
2.  A list of currently open tabs.
3.  A textual representation of interactive elements on the page, each with an index.
4.  A screenshot of the current page (if vision is enabled).
5.  The results or errors from your previous action(s).

Your response MUST be a single tool call to the 'NextAction' tool.
The arguments for 'NextAction' must be a JSON object with two main keys:
    -   "current_state": An object containing your analysis of the current situation.
        -   "evaluation_previous_goal": Briefly evaluate the outcome of your last action(s) (e.g., "Success, found the item", "Failed, element not interactable", "Unknown, page loaded but need to verify").
        -   "memory": Concisely summarize what has been achieved so far and any critical information gathered that is relevant to the overall task. This helps maintain context.
        -   "next_goal": Clearly state your immediate next sub-goal or the question you are trying to answer with the next action(s).
    -   "action": A list of one or more actions to be performed. Each action in the list is an object with a single key, where the key is the action name and the value is an object of its parameters.

Available actions (use these as keys in the "action" list items):
{action_description}

General Guidelines:
-   Be methodical. Break down complex tasks into smaller, manageable steps.
-   If a page is long, use scroll actions to explore. Elements not visible in the screenshot might require scrolling.
-   If an action fails, analyze the error and the current page state to decide on a recovery action or a different approach.
-   Pay attention to element indices. Use the correct index for the element you intend to interact with.
-   If the task requires extracting specific information, use the 'extract_content' action with a clear goal.
-   When the task is fully completed, use the 'done' action. Set 'success' to true if the task was achieved, or false if it could not be completed as requested. Provide a summary in the 'text' field of the 'done' action.
-   You can perform up to {max_actions_per_step} actions in a single step if it makes sense (e.g., typing then clicking submit). List them sequentially in the "action" list.
-   If elements are not found, consider if the page is still loading, if you need to scroll, or if the previous action led to an unexpected page.
"""


class SystemPrompt:
    def __init__(
        self,
        action_description: str,
        max_actions_per_step: int = 3,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
    ):
        prompt_content = override_system_message or PROMPT_TEMPLATE.format(
            action_description=action_description,
            max_actions_per_step=max_actions_per_step,
        )
        if extend_system_message:
            prompt_content += f"\n{extend_system_message}"

        self.system_message = SystemMessage(content=prompt_content)

    def get_system_message(self) -> SystemMessage:
        return self.system_message


class AgentMessagePrompt:
    def __init__(
        self,
        browser_state_summary: BrowserStateSummary,
        result: list[StepResult] | None = None,
        include_attributes: list[str] | None = None,
        step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = browser_state_summary
        self.result = result
        self.include_attributes = include_attributes or []
        self.step_info = step_info
        assert self.state is not None, "BrowserStateSummary cannot be None"

    def get_user_message_text_part(self) -> str:
        elements_text = (
            self.state.clickable_elements_to_string(
                include_attributes=self.include_attributes
            )
            if self.state.element_tree or self.state.selector_map
            else "No interactive elements data available or empty page."
        )

        has_content_above = (self.state.pixels_above or 0) > 50
        has_content_below = (self.state.pixels_below or 0) > 50

        scroll_info = []
        if has_content_above:
            scroll_info.append(
                f"... Content continues for approx. {self.state.pixels_above}px above ..."
            )
        else:
            scroll_info.append("[Top of viewable page area]")

        if has_content_below:
            scroll_info.append(
                f"... Content continues for approx. {self.state.pixels_below}px below ..."
            )
        else:
            scroll_info.append("[Bottom of viewable page area]")

        elements_text_with_scroll = (
            f"{scroll_info[0]}\n{elements_text}\n{scroll_info[1]}"
        )

        step_info_desc = ""
        if self.step_info:
            step_info_desc = f"Current Step: {self.step_info.step_number + 1} of {self.step_info.max_steps}. "
        step_info_desc += (
            f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        tabs_fmt = (
            "\n".join(
                [
                    f"- Tab {t.get('page_id', 'N/A')}: '{str(t.get('title', 'No Title'))[:40]}' ({str(t.get('url', 'No URL'))[:50]})"
                    for t in self.state.tabs
                ]
            )
            if self.state.tabs
            else "No other tabs open or tab information unavailable."
        )

        state_desc = (
            f"\n[End of Prior Action Results]\n\n[Current Page State]\n"
            f"URL: {self.state.url}\n"
            f"Title: {self.state.title}\n"
            f"Open Tabs:\n{tabs_fmt}\n\n"
            f"Interactive Elements Visible on Page (scroll if not listed):\n{elements_text_with_scroll}\n\n"
            f"{step_info_desc}\n"
            f"Please provide your 'current_state' analysis and next 'action'(s)."
        )
        return state_desc

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        state_desc_text = self.get_user_message_text_part()

        if self.state.screenshot and use_vision:
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_desc_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.state.screenshot}"
                        },
                    },
                ]
            )
        else:
            return HumanMessage(content=state_desc_text)


class AgentBrain(BaseModel):
    evaluation_previous_goal: str = Field(
        ...,
        description="Brief evaluation of the last action's outcome (Success/Failed/Unknown, and key reason/observation).",
    )
    memory: str = Field(
        ...,
        description="Concise summary of progress towards the main task and critical info found so far.",
    )
    next_goal: str = Field(
        ..., description="The immediate, specific sub-goal for the next action(s)."
    )


class NextAction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_state: AgentBrain = Field(
        ..., description="WebAutomator's analysis of the current situation."
    )
    action: List[ActionModel] = Field(
        ..., min_length=1, description="List of one or more actions to perform."
    )

    @staticmethod
    def type_with_custom_actions(
        custom_actions_model: Type[ActionModel],
    ) -> Type["NextAction"]:
        return create_model(
            "DynamicAgentOutput",
            __base__=NextAction,
            action=(List[custom_actions_model], Field(..., min_length=1)),
            __module__=NextAction.__module__,
        )


class AutomationConfig(BaseModel):
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
    max_actions_per_step: int = 3
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


class AutomatorStatus(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_steps: int = 0
    consecutive_failures: int = 0
    last_result: Optional[List[StepResult]] = None
    history: "AgentHistoryList" = Field(
        default_factory=lambda: AgentHistoryList(history=[])
    )
    message_manager_state: MessageManagerState = Field(
        default_factory=MessageManagerState
    )
    paused: bool = False
    stopped: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExecutionLog(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_output: Optional[NextAction] = None
    result: List[StepResult]
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AgentHistoryList(BaseModel):
    history: List[ExecutionLog] = Field(default_factory=list)
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


AutomatorStatus.model_rebuild()


class WebAutomator(Generic[Context]):
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: Optional[WebNavigator] = None,
        controller: Optional[ActionManager[Context]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        context: Optional[Context] = None,
        agent_settings: Optional[AutomationConfig] = None,
        browser_profile: Optional[BrowserConfig] = None,
        injected_agent_state: Optional[AutomatorStatus] = None,
    ):
        self.task = task
        self.llm = llm
        self.controller = controller or ActionManager()
        self.sensitive_data = sensitive_data
        self.version = "main.py-refactored-0.8"
        self.settings = agent_settings or AutomationConfig()
        self.state = injected_agent_state or AutomatorStatus()

        self.browser_session = browser_session or WebNavigator(
            browser_profile=(browser_profile or DEFAULT_BROWSER_PROFILE)
        )
        self.context = context

        self.ActionModelType = self.controller.registry.create_action_model()
        self.AgentOutputType = NextAction.type_with_custom_actions(self.ActionModelType)

        self.initial_actions = (
            self._convert_initial_actions(initial_actions) if initial_actions else None
        )

        self.tool_calling_method = self.settings.tool_calling_method
        if self.tool_calling_method == "auto":
            if (
                "openai" in self.llm.__class__.__name__.lower()
                or "azure" in self.llm.__class__.__name__.lower()
            ):
                self.tool_calling_method = "tools"
            elif "anthropic" in self.llm.__class__.__name__.lower():
                self.tool_calling_method = "tools"
            elif "google" in self.llm.__class__.__name__.lower():
                self.tool_calling_method = "tools"
            else:
                logger.warning(
                    f"Tool calling method 'auto' for {self.llm.__class__.__name__}, defaulting to 'tools'. May need explicit setting."
                )
                self.tool_calling_method = "tools"

        if not hasattr(self.llm, "bind_tools") and self.tool_calling_method not in [
            "raw",
            None,
            "json_mode",
        ]:
            logger.warning(
                f"LLM {self.llm.__class__.__name__} may not support bind_tools with method '{self.tool_calling_method}'. "
                "Tool calling might fail. Consider 'raw' or 'json_mode' if issues arise."
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
        converted = []
        for action_dict in actions:
            if self._validate_initial_action(action_dict):
                try:
                    action_name = list(action_dict.keys())[0]
                    action_params = action_dict[action_name]

                    action_instance_params = {action_name: action_params}
                    converted.append(self.ActionModelType(**action_instance_params))
                except (ValidationError, IndexError, TypeError) as e:
                    logger.error(f"Error converting initial action {action_dict}: {e}")
            else:
                logger.warning(f"Skipping invalid initial action: {action_dict}")
        return converted

    def _validate_initial_action(self, action_dict: Dict[str, Dict[str, Any]]) -> bool:
        if not isinstance(action_dict, dict) or len(action_dict) != 1:
            logger.error(
                f"Initial action format error: Each action must be a dict with a single key (action_name). Got: {action_dict}"
            )
            return False
        action_name = list(action_dict.keys())[0]
        if action_name not in self.controller.registry.actions:
            logger.error(
                f"Initial action '{action_name}' not found in registered actions."
            )
            return False

        param_model = self.controller.registry.actions[action_name].param_model
        action_params = action_dict[action_name]
        try:
            param_model(**action_params)
            return True
        except ValidationError as e:
            logger.error(
                f"Failed to validate parameters for initial action '{action_name}': {e}. Params: {action_params}"
            )
            return False

    async def get_next_action(
        self, input_messages: List[BaseMessage]
    ) -> Tuple[NextAction, Optional[str]]:
        llm_to_invoke = self.llm
        tool_call_id_from_llm: Optional[str] = None

        agent_output_tool_name = self.AgentOutputType.__name__

        try:
            if self.tool_calling_method in ["tools", "function_calling"]:
                llm_to_invoke = self.llm.bind_tools(
                    tools=[self.AgentOutputType],
                    tool_choice={
                        "type": "function",
                        "function": {"name": agent_output_tool_name},
                    },
                )
            elif self.tool_calling_method == "json_mode":
                llm_to_invoke = self.llm.with_structured_output(
                    self.AgentOutputType, method="json_mode", include_raw=True
                )

        except Exception as e:
            logger.error(
                f"Failed to configure LLM for tool/structured output with method '{self.tool_calling_method}': {e}"
            )

        raw_response_message: BaseMessage = await llm_to_invoke.ainvoke(input_messages)
        model_output_obj: NextAction

        if (
            hasattr(raw_response_message, "tool_calls")
            and raw_response_message.tool_calls
        ):
            tool_call = raw_response_message.tool_calls[0]
            tool_call_id_from_llm = tool_call.get("id")
            called_tool_name = tool_call.get("name")
            raw_args = tool_call.get("args")

            if called_tool_name != agent_output_tool_name:
                raise ValueError(
                    f"LLM called unexpected tool: '{called_tool_name}'. Expected '{agent_output_tool_name}'."
                )

            try:
                args_dict = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
                model_output_obj = self.AgentOutputType(**args_dict)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(
                    f"Failed to parse/validate NextAction from tool args: {e}. Raw Args: {raw_args}"
                )
                raise ValueError(
                    f"Invalid arguments for tool '{agent_output_tool_name}': {e}. Args: {raw_args}"
                ) from e

        elif (
            self.tool_calling_method == "json_mode"
            and isinstance(raw_response_message, dict)
            and "parsed" in raw_response_message
        ):
            if isinstance(raw_response_message["parsed"], self.AgentOutputType):
                model_output_obj = raw_response_message["parsed"]
            else:
                raw_content_for_fallback = raw_response_message.get("raw", {}).get(
                    "content", ""
                )
                try:
                    model_output_obj = self.AgentOutputType(
                        **json.loads(raw_content_for_fallback)
                    )
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(
                        f"Fallback JSON parsing failed for json_mode: {e}. Raw content: {raw_content_for_fallback}"
                    )
                    raise ValueError(f"Could not parse json_mode output: {e}") from e

        elif isinstance(raw_response_message.content, str):
            content_str = raw_response_message.content.strip()
            if content_str.startswith("```json"):
                content_str = content_str[7:]
            if content_str.endswith("```"):
                content_str = content_str[:-3]
            content_str = content_str.strip()
            try:
                model_output_obj = self.AgentOutputType(**json.loads(content_str))
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(
                    f"Failed to parse LLM string content as NextAction: {e}. Content: {raw_response_message.content}"
                )
                raise ValueError(
                    f"LLM response content could not be parsed into NextAction structure: {e}. Content: {content_str}"
                ) from e
        else:
            raise ValueError(
                f"LLM response in unexpected format: {type(raw_response_message)}. Full response: {raw_response_message}"
            )

        log_response(model_output_obj)
        return model_output_obj, tool_call_id_from_llm

    async def multi_act(self, actions: List[ActionModel]) -> List[StepResult]:
        results: List[StepResult] = []

        initial_hashes_url = None
        initial_hashes_set = set()
        if self.browser_session._cached_browser_state_summary:
            initial_hashes_url = self.browser_session._cached_browser_state_summary.url
            if self.browser_session._cached_clickable_element_hashes:
                initial_hashes_set = (
                    self.browser_session._cached_clickable_element_hashes.get(
                        "hashes", set()
                    )
                )

        for i, action_model_instance in enumerate(actions):
            if self.state.stopped:
                results.append(
                    StepResult(
                        error="WebAutomator stopped during action sequence.",
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
                        )
                    except ValidationError as e:
                        error_msg = f"Invalid action structure for action {i + 1} in sequence: {e}. Action data: {current_action_model}"
                        logger.error(error_msg)
                        results.append(StepResult(error=error_msg))
                        break
                else:
                    error_msg = f"Action {i + 1} in sequence is not a valid ActionModel or dictionary. Got type: {type(current_action_model)}"
                    logger.error(error_msg)
                    results.append(StepResult(error=error_msg))
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

                try:
                    fresh_state_summary = await self.browser_session.get_state_summary(
                        cache_clickable_elements_hashes=True
                    )
                    current_url_after_action = fresh_state_summary.url
                    new_hashes = (
                        self.browser_session._cached_clickable_element_hashes.get(
                            "hashes", set()
                        )
                        if self.browser_session._cached_clickable_element_hashes
                        else set()
                    )

                    url_changed = current_url_after_action != initial_hashes_url
                    dom_significantly_changed = (
                        not url_changed and new_hashes != initial_hashes_set
                    ) or (url_changed and new_hashes != initial_hashes_set)

                    if url_changed:
                        logger.info(
                            f"URL changed from '{initial_hashes_url}' to '{current_url_after_action}' after action {i + 1}. Interrupting multi-action step for re-evaluation."
                        )
                        results.append(
                            StepResult(
                                extracted_content="Page URL changed, re-evaluating.",
                                include_in_memory=True,
                            )
                        )
                        break
                    if dom_significantly_changed:
                        logger.info(
                            f"DOM structure significantly changed on URL '{initial_hashes_url}' after action {i + 1}. Interrupting multi-action step for re-evaluation."
                        )
                        results.append(
                            StepResult(
                                extracted_content="DOM changed, re-evaluating.",
                                include_in_memory=True,
                            )
                        )
                        break

                    initial_hashes_url = current_url_after_action
                    initial_hashes_set = new_hashes

                except Exception as e:
                    logger.warning(
                        f"Error during DOM change check in multi_act: {e}. Continuing with next action."
                    )

            if i < len(actions) - 1:
                await asyncio.sleep(
                    self.browser_session.browser_profile.wait_between_actions
                )
        return results

    async def _handle_step_error(self, error: Exception) -> List[StepResult]:
        error_msg = f"{type(error).__name__}: {str(error)}"
        logger.error(f"Step execution failed: {error_msg}", exc_info=True)
        self.state.consecutive_failures += 1
        return [StepResult(error=error_msg, include_in_memory=True)]

    async def step(self, step_info: Optional[AgentStepInfo] = None):
        self.state.n_steps += 1
        logger.info(f"--- Step {self.state.n_steps} ---")

        browser_state_summary: Optional[BrowserStateSummary] = None
        model_decision_output: Optional[NextAction] = None
        action_execution_results: List[StepResult] = []
        tool_call_id_for_this_step: Optional[str] = None

        try:
            if not self.browser_session.initialized:
                await self.browser_session.start()

            browser_state_summary = await self.browser_session.get_state_summary(
                cache_clickable_elements_hashes=True
            )

            self._message_manager.add_state_message(
                browser_state_summary=browser_state_summary,
                result=self.state.last_result,
                step_info=step_info,
                use_vision=self.settings.use_vision,
            )

            input_messages_for_llm = self._message_manager.get_messages()

            (
                model_decision_output,
                tool_call_id_for_this_step,
            ) = await self.get_next_action(input_messages_for_llm)

            self._message_manager.state.history.remove_last_state_message()

            if model_decision_output and tool_call_id_for_this_step:
                self._message_manager.add_model_output(
                    model_decision_output, tool_call_id_for_this_step
                )
            elif model_decision_output:
                fallback_tc_id = f"ftc_{self.state.n_steps}_{str(uuid.uuid4())[:4]}"
                self._message_manager.add_model_output(
                    model_decision_output, fallback_tc_id
                )
                tool_call_id_for_this_step = fallback_tc_id
                logger.warning(
                    "Model output processed without a tool_call_id from LLM, using fallback."
                )

            if model_decision_output and model_decision_output.action:
                action_execution_results = await self.multi_act(
                    model_decision_output.action
                )
            else:
                logger.warning("LLM decided no actions or action list was empty.")
                action_execution_results = [
                    StepResult(error="LLM provided no actions.", include_in_memory=True)
                ]

            self.state.last_result = action_execution_results
            self.state.consecutive_failures = 0

            if tool_call_id_for_this_step:
                combined_summary = "; ".join(
                    f"Action {idx + 1}: {(r.extracted_content or r.error or 'OK')[:100]}"
                    for idx, r in enumerate(action_execution_results)
                )
                self._message_manager.add_tool_message(
                    content=f"Executed actions. Summary: {combined_summary if combined_summary else 'No specific result summary.'}",
                    tool_call_id=tool_call_id_for_this_step,
                )
            else:
                logger.error(
                    "Cannot add ToolMessage: tool_call_id for the step was not established."
                )

        except Exception as e:
            action_execution_results = await self._handle_step_error(e)
            self.state.last_result = action_execution_results
            if tool_call_id_for_this_step:
                self._message_manager.add_tool_message(
                    content=f"Critical error during step execution: {str(e)}",
                    tool_call_id=tool_call_id_for_this_step,
                )
            else:
                self._message_manager._add_message_with_tokens(
                    HumanMessage(content=f"[AGENT ERROR] Step failed: {str(e)}")
                )
                logger.error(
                    f"Error in step before tool_call_id was obtained or for non-tool_call flow: {e}"
                )

        finally:
            if browser_state_summary and model_decision_output:
                self.state.history.history.append(
                    ExecutionLog(
                        model_output=model_decision_output,
                        result=action_execution_results,
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
            elif (
                browser_state_summary
                and not model_decision_output
                and action_execution_results
            ):
                self.state.history.history.append(
                    ExecutionLog(
                        model_output=None,
                        result=action_execution_results,
                        state={
                            "url": browser_state_summary.url,
                            "title": browser_state_summary.title,
                            "screenshot_summary": (
                                browser_state_summary.screenshot[:100]
                                if browser_state_summary.screenshot
                                else None
                            ),
                        },
                        metadata={"step": self.state.n_steps, "error_before_llm": True},
                    )
                )

    async def start_task(
        self,
        max_steps: int = 100,
        on_step_start: Optional[Callable[["WebAutomator"], Awaitable[None]]] = None,
        on_step_end: Optional[Callable[["WebAutomator"], Awaitable[None]]] = None,
    ) -> AgentHistoryList:
        logger.info(f"🚀 Starting task: {self.task} (WebAutomator v{self.version})")

        if self.initial_actions:
            logger.info(f"Executing initial actions: {self.initial_actions}")
            self.state.last_result = await self.multi_act(self.initial_actions)
            if self.state.last_result:
                current_bs_state = None
                if self.browser_session.initialized:
                    try:
                        current_bs_state = await self.browser_session.get_state_summary(
                            cache_clickable_elements_hashes=False
                        )
                    except Exception as bse:
                        logger.warning(
                            f"Could not get browser state after initial actions: {bse}"
                        )

                self.state.history.history.append(
                    ExecutionLog(
                        model_output=None,
                        result=self.state.last_result,
                        state={
                            "url": current_bs_state.url if current_bs_state else "N/A",
                            "title": current_bs_state.title
                            if current_bs_state
                            else "N/A",
                            "note": "State after initial actions",
                        },
                        metadata={"step": 0, "type": "initial_actions"},
                    )
                )

        for step_num_zero_indexed in range(max_steps):
            actual_step_num = self.state.n_steps + 1

            if self.state.stopped:
                logger.info("WebAutomator stopped.")
                break

            if self.state.paused:
                logger.info("WebAutomator paused. Waiting for resume...")
                await self._external_pause_event.wait()
                if self.state.stopped:
                    logger.info("WebAutomator stopped during pause.")
                    break
                logger.info("WebAutomator resumed.")

            if self.state.consecutive_failures >= self.settings.max_failures:
                logger.error(
                    f"Stopping due to {self.settings.max_failures} consecutive failures."
                )
                if not self.state.history.is_done():
                    self._add_failure_done_action("Max consecutive failures reached.")
                break

            current_step_info = AgentStepInfo(
                step_number=actual_step_num - 1, max_steps=max_steps
            )

            if on_step_start:
                await on_step_start(self)

            await self.step(step_info=current_step_info)

            if on_step_end:
                await on_step_end(self)

            if self.state.history.is_done():
                logger.info("✅ Task marked as done by agent.")
                break
        else:
            if not self.state.history.is_done():
                logger.info(f"Max steps ({max_steps}) reached. Task may be incomplete.")
                self._add_failure_done_action(
                    "Max steps reached, task not marked done by agent."
                )

        logger.info(
            f"WebAutomator run finished. Total agent steps executed: {self.state.n_steps}"
        )
        if self.settings.save_conversation_path:
            self._save_conversation()

        return self.state.history

    def _add_failure_done_action(self, reason: str):
        logger.warning(f"Forcing task completion as failure: {reason}")

        if self.state.history.history:
            last_hist_item = self.state.history.history[-1]
            if (
                last_hist_item.result
                and last_hist_item.result[-1].is_done
                and not last_hist_item.result[-1].success
                and reason in (last_hist_item.result[-1].extracted_content or "")
            ):
                logger.info(
                    f"Failure reason '{reason}' already matches the last 'done' action. Not adding duplicate."
                )
                return

        done_params = DoneAction(text=reason, success=False)
        try:
            final_action = self.ActionModelType(**{"done": done_params})
        except ValidationError as ve:
            logger.error(
                f"Could not create 'done' ActionModel: {ve}. This might indicate a mismatch in ActionModelType structure."
            )
            if hasattr(self.ActionModelType, "done"):
                final_action = self.ActionModelType(done=done_params)
            else:
                logger.error(
                    "Cannot determine how to structure 'done' action for ActionModelType. Skipping forced done history item."
                )
                return

        final_brain_state = AgentBrain(
            evaluation_previous_goal=f"Task aborted: {reason}",
            memory="Task could not be completed successfully.",
            next_goal="N/A - Task ended due to failure condition.",
        )

        final_model_output = self.AgentOutputType(
            current_state=final_brain_state,
            action=[final_action],
        )

        final_action_result = StepResult(
            is_done=True, success=False, extracted_content=reason
        )

        self.state.history.history.append(
            ExecutionLog(
                model_output=final_model_output,
                result=[final_action_result],
                state={
                    "url": "N/A",
                    "title": "N/A",
                    "note": "Forced completion due to failure/timeout",
                },
                metadata={
                    "step": self.state.n_steps,
                    "reason": f"Forced done: {reason}",
                },
            )
        )
        logger.info(
            f"Failure 'done' action added to history for step {self.state.n_steps}."
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

            with open(file_path_str, "w", encoding="utf-8") as f:
                f.write(self.state.history.model_dump_json(indent=2))
            logger.info(f"WebAutomator history saved to {file_path_str}")
        except Exception as e:
            logger.error(
                f"Failed to save conversation to {self.settings.save_conversation_path}: {e}",
                exc_info=True,
            )

    async def close(self):
        if self.browser_session and self.browser_session.initialized:
            await self.browser_session.stop()
        logger.info("WebAutomator closed and browser session stopped.")

    def pause(self):
        self.state.paused = True
        self._external_pause_event.clear()
        logger.info(
            "WebAutomator execution paused. Call resume() to continue or stop() to terminate."
        )

    def resume(self):
        if self.state.paused:
            self.state.paused = False
            self._external_pause_event.set()
            logger.info("WebAutomator execution resumed.")
        else:
            logger.info("WebAutomator is not paused.")

    def stop(self):
        self.state.stopped = True
        if self.state.paused:
            self._external_pause_event.set()
        logger.info(
            "WebAutomator stop requested. The current step will attempt to complete if safe."
        )


def log_response(response: NextAction) -> None:
    emoji = "🤷"
    if response.current_state:
        if response.current_state.evaluation_previous_goal:
            eval_text = response.current_state.evaluation_previous_goal.lower()
            if (
                "success" in eval_text
                or "found" in eval_text
                or "completed" in eval_text
            ):
                emoji = "👍"
            elif "fail" in eval_text or "error" in eval_text or "unable" in eval_text:
                emoji = "⚠️"
            logger.info(
                f"{emoji} LLM Eval: {response.current_state.evaluation_previous_goal}"
            )
        logger.info(f"🧠 LLM Memory: {response.current_state.memory}")
        logger.info(f"🎯 LLM Next Goal: {response.current_state.next_goal}")

    if response.action:
        for i, action_instance in enumerate(response.action):
            action_dict_for_log = action_instance.model_dump(
                exclude_unset=True, exclude_none=True
            )

            if action_dict_for_log:
                action_name = list(action_dict_for_log.keys())[0]
                action_params = action_dict_for_log[action_name]
                try:
                    params_str = json.dumps(action_params)
                except TypeError:
                    params_str = str(action_params)

                logger.info(
                    f"🛠️ LLM Action {i + 1}/{len(response.action)}: {action_name}({params_str})"
                )
            else:
                logger.warning(
                    f"LLM Action {i + 1}/{len(response.action)} is empty or invalid after model_dump."
                )
    else:
        logger.warning("LLM NextAction contains no actions.")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not found.")
        logger.error("OPENAI_API_KEY not found.")
        return

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    task = "Go to google.com, search for 'current weather in Ho Chi Minh City', and then extract the temperature and general conditions."

    persistent_profile_path = os.getenv("PERSISTENT_PROFILE_PATH")
    chrome_exe_path = os.getenv("CHROME_EXE_PATH")

    current_bp = DEFAULT_BROWSER_PROFILE.model_copy(
        update={
            "user_data_dir": persistent_profile_path,
            "executable_path": chrome_exe_path,
            "headless": False,
            "args": (DEFAULT_BROWSER_PROFILE.args or [])
            + ["--start-maximized", "--disable-gpu", "--no-sandbox"],
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
            "wait_between_actions": 1.0,
            "viewport": {
                "width": 1920,
                "height": 1080,
            },
        }
    )

    conversation_dir = Path("conversations")
    conversation_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_as = AutomationConfig(
        use_vision=True,
        max_actions_per_step=2,
        tool_calling_method="tools",
        page_extraction_llm=llm,
        save_conversation_path=str(
            conversation_dir / f"{timestamp}_agent_run_history.json"
        ),
        max_failures=2,
    )

    agent = WebAutomator(
        task=task,
        llm=llm,
        agent_settings=current_as,
        browser_profile=current_bp,
        controller=ActionManager(),
    )

    try:
        logger.info(f"🚀 Starting agent task: {task}")
        history: AgentHistoryList = await agent.start_task(max_steps=10)

        print("\n--- WebAutomator Run History Summary ---")
        if history.history:
            for i, hist_item in enumerate(history.history):
                step_metadata = hist_item.metadata or {}
                print(
                    f"\n--- History Record for WebAutomator Step ~{step_metadata.get('step', i + 1)} ---"
                )

                if hist_item.model_output:
                    cs = hist_item.model_output.current_state
                    print(
                        f"  LLM Decided -> Eval: '{cs.evaluation_previous_goal}' | Memory: '{cs.memory}' | Next Goal: '{cs.next_goal}'"
                    )
                    for action_idx, action_item in enumerate(
                        hist_item.model_output.action
                    ):
                        ad = action_item.model_dump(
                            exclude_unset=True, exclude_none=True
                        )
                        if ad:
                            action_name = list(ad.keys())[0]
                            action_params = json.dumps(list(ad.values())[0])
                            print(
                                f"    LLM Action {action_idx + 1}: {action_name}({action_params})"
                            )
                else:
                    print(
                        "  No LLM output for this history item (e.g., initial actions or forced failure)."
                    )

                if hist_item.result:
                    for res_idx, res_item in enumerate(hist_item.result):
                        print(f"  Actual Action Result {res_idx + 1}:")
                        if res_item.extracted_content:
                            print(
                                f"    Content: {res_item.extracted_content[:300].strip()}..."
                            )
                        if res_item.error:
                            print(f"    Error: {res_item.error}")
                        if res_item.is_done:
                            print(f"    Task Marked Done: Success={res_item.success}")

                browser_s = hist_item.state or {}
                if browser_s.get("url"):
                    print(
                        f"  Browser State Context: URL='{browser_s.get('url')}', Title='{browser_s.get('title', 'N/A')}'"
                    )
                if step_metadata.get("note"):
                    print(f"  Note: {step_metadata.get('note')}")
                if step_metadata.get("reason"):
                    print(f"  Reason: {step_metadata.get('reason')}")

        final_content = history.final_result()
        if final_content:
            print(f"\n✅ Final Result from WebAutomator: {final_content}")
        else:
            if history.is_done() and not history.is_successful():
                print("\n❌ WebAutomator marked task as done, but reported failure.")
            else:
                print(
                    "\n❓ WebAutomator did not complete the task successfully or produce a final result within the step limit."
                )

    except Exception as e:
        print(f"An error occurred during agent execution: {type(e).__name__} - {e}")
        logger.error("Main execution loop error", exc_info=True)
    finally:
        print("Closing browser session...")


def __main__():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())


if __name__ == "__main__":
    __main__()