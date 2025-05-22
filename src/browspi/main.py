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
from enum import Enum
from functools import wraps
from pathlib import Path
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

# Original Playwright imports for non-TypedDict types or those not directly in Pydantic models
from playwright.async_api import (
    Browser as PlaywrightBrowser,
)
from playwright.async_api import (
    BrowserContext as PlaywrightBrowserContext,
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
from playwright.async_api import async_playwright as playwright_async_playwright

# Note: Types from _api_structures (ClientCertificate, Geolocation, HttpCredentials)
# will be effectively replaced by our PydanticCompatible versions for model definitions.
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    create_model,
    model_validator,
)

# Ensure TypedDict and NotRequired are imported from typing_extensions
from typing_extensions import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    NotRequired,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
)  # Added TypedDict, NotRequired

from browspi.dom.views import (
    DOMElementNode,
    DOMState,
    SelectorMap,
)
from browspi.dom_utils.service import DomService

# Suppress specific LangChain Beta warning
filterwarnings("ignore", category=LangChainBetaWarning)

# Load environment variables
load_dotenv()

# --- Logging Configuration (remains unchanged) ---
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

    class BrowserUseFormatter(logging.Formatter):
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
        console.setFormatter(BrowserUseFormatter("%(message)s"))
    else:
        console.setFormatter(
            BrowserUseFormatter("%(levelname)-8s [%(name)s] %(message)s")
        )
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


# --- Utility Functions (remains unchanged) ---
def time_execution_sync(
    additional_text: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{additional_text} Execution time: {execution_time:.2f} seconds"
            )
            return result

        return wrapper

    return decorator


def time_execution_async(
    additional_text: str = "",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{additional_text} Execution time: {execution_time:.2f} seconds"
            )
            return result

        return wrapper

    return decorator


def check_env_variables(keys: List[str], any_or_all=all) -> bool:
    return any_or_all(os.getenv(key, "").strip() for key in keys)


# --- Redefined Playwright TypedDicts for Pydantic compatibility ---
class PydanticCompatibleProxySettings(TypedDict, total=False):
    server: str
    bypass: str
    username: str
    password: str


class PydanticCompatibleViewportSize(TypedDict):
    width: int
    height: int


class PydanticCompatibleGeolocation(TypedDict, total=False):
    latitude: float
    longitude: float
    accuracy: float


class PydanticCompatibleHttpCredentials(TypedDict):
    username: str
    password: str


class PydanticCompatibleClientCertificate(TypedDict):  # total=True (default)
    origin: str
    path: Union[str, Path]
    password: NotRequired[str]


# --- Pydantic Models for Browser Configuration ---
IN_DOCKER = os.environ.get("IN_DOCKER", "false").lower()[0] in "ty1"
CHROME_HEADLESS_ARGS = ["--headless=new"]
CHROME_DOCKER_ARGS = [
    "--no-sandbox",
    "--disable-gpu-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--no-xshm",
    "--no-zygote",
    "--single-process",
]
CHROME_DEFAULT_ARGS = [
    "--disable-field-trial-config",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-back-forward-cache",
    "--disable-breakpad",
    "--disable-client-side-phishing-detection",
    "--disable-component-extensions-with-background-pages",
    "--disable-component-update",
    "--no-default-browser-check",
    "--disable-dev-shm-usage",
    "--allow-pre-commit-input",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-renderer-backgrounding",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store=basic",
    "--use-mock-keychain",
    "--no-service-autorun",
    "--export-tagged-pdf",
    "--disable-search-engine-choice-screen",
    "--unsafely-disable-devtools-self-xss-warnings",
    "--enable-features=NetworkService,NetworkServiceInProcess",
    "--enable-network-information-downlink-max",
]


class ColorScheme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    NO_PREFERENCE = "no-preference"


class BrowserContextArgs(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=False,
        revalidate_instances="always",
        populate_by_name=True,
    )
    accept_downloads: bool = True
    proxy: Optional[PydanticCompatibleProxySettings] = None  # MODIFIED
    user_agent: Optional[str] = None
    viewport: Optional[PydanticCompatibleViewportSize] = Field(default=None)  # MODIFIED
    no_viewport: Optional[bool] = None
    locale: Optional[str] = None
    geolocation: Optional[PydanticCompatibleGeolocation] = None  # MODIFIED
    timezone_id: Optional[str] = None
    color_scheme: ColorScheme = ColorScheme.LIGHT
    extra_http_headers: Dict[str, str] = Field(default_factory=dict)
    offline: bool = False
    http_credentials: Optional[PydanticCompatibleHttpCredentials] = None  # MODIFIED
    ignore_https_errors: bool = False
    java_script_enabled: bool = True
    bypass_csp: bool = False
    service_workers: Literal["allow", "block"] = "allow"
    client_certificates: List[PydanticCompatibleClientCertificate] = Field(
        default_factory=list
    )  # MODIFIED
    record_har_path: Optional[Union[str, Path]] = None
    record_video_dir: Optional[Union[str, Path]] = None
    record_video_size: Optional[PydanticCompatibleViewportSize] = None  # MODIFIED
    strict_selectors: bool = False


class BrowserLaunchArgs(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        revalidate_instances="always",
        from_attributes=True,
        populate_by_name=True,
    )
    executable_path: Optional[Union[str, Path]] = Field(
        default=None,
        validation_alias=AliasChoices("chrome_binary_path", "browser_binary_path"),
    )
    headless: Optional[bool] = Field(default=None)
    args: List[str] = Field(default_factory=list)
    ignore_default_args: Union[List[str], Literal[True]] = Field(
        default_factory=lambda: ["--enable-automation", "--disable-extensions"]
    )
    channel: Optional[str] = "chromium"
    chromium_sandbox: bool = Field(default=not IN_DOCKER)
    devtools: bool = Field(default=False)
    env: Dict[str, Union[str, float, bool]] = Field(default_factory=dict)
    handle_sighup: bool = True
    handle_sigint: bool = False
    handle_sigterm: bool = False
    proxy: Optional[PydanticCompatibleProxySettings] = None  # MODIFIED
    slow_mo: float = 0.0
    timeout: float = 30000.0
    traces_dir: Optional[Union[str, Path]] = None
    downloads_path: Optional[Union[str, Path]] = None

    @model_validator(mode="after")
    def validate_devtools_headless(self) -> "BrowserLaunchArgs":
        if self.headless and self.devtools:
            raise ValueError(
                "headless=True and devtools=True cannot both be set at the same time"
            )
        return self

    @staticmethod
    def args_as_dict(args: list[str]) -> dict[str, str]:
        args_dict = {}
        for arg in args:
            key_value = arg.split("=", 1)
            key = key_value[0].strip().lstrip("-")
            value = key_value[1].strip() if len(key_value) > 1 else ""
            args_dict[key] = value
        return args_dict

    @staticmethod
    def args_as_list(args: dict[str, str]) -> list[str]:
        return [
            f"--{key.lstrip('-')}={value}" if value else f"--{key.lstrip('-')}"
            for key, value in args.items()
        ]


class BrowserNewContextArgs(BrowserContextArgs):  # Inherits changes
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=False,
        revalidate_instances="always",
        populate_by_name=True,
    )
    storage_state: Optional[Union[str, Path, Dict[str, Any]]] = (
        None  # Note: PlaywrightStorageState could also be redefined if used directly
    )


class BrowserLaunchPersistentContextArgs(
    BrowserLaunchArgs, BrowserContextArgs
):  # Inherits changes
    model_config = ConfigDict(
        extra="ignore", validate_assignment=False, revalidate_instances="always"
    )
    user_data_dir: Optional[Union[str, Path]] = Path(
        "~/.config/browseruse/profiles/default"
    ).expanduser()


class BrowserProfile(
    BrowserLaunchPersistentContextArgs, BrowserLaunchArgs, BrowserNewContextArgs
):  # Inherits changes
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        revalidate_instances="always",
        from_attributes=True,
        validate_by_name=True,
        populate_by_name=True,
    )
    disable_security: bool = Field(default=False)
    deterministic_rendering: bool = Field(default=False)
    allowed_domains: Optional[List[str]] = Field(default=None)
    keep_alive: Optional[bool] = Field(default=None)
    window_size: Optional[PydanticCompatibleViewportSize] = Field(
        default=None
    )  # MODIFIED
    minimum_wait_page_load_time: float = Field(default=0.25)
    wait_for_network_idle_page_load_time: float = Field(default=0.5)
    maximum_wait_page_load_time: float = Field(default=5.0)
    wait_between_actions: float = Field(default=0.5)
    include_dynamic_attributes: bool = Field(default=True)
    highlight_elements: bool = Field(default=True)
    viewport_expansion: int = Field(default=500)
    cookies_file: Optional[str] = Field(default=None)
    profile_directory: str = "Default"

    def get_args(self) -> list[str]:
        current_args = list(self.args)
        if self.headless:
            current_args.extend(CHROME_HEADLESS_ARGS)
        if IN_DOCKER:
            current_args.extend(CHROME_DOCKER_ARGS)
        current_args.extend(CHROME_DEFAULT_ARGS)
        return sorted(list(set(current_args)), key=current_args.index)

    def kwargs_for_launch_persistent_context(
        self,
    ) -> BrowserLaunchPersistentContextArgs:
        return BrowserLaunchPersistentContextArgs(
            **self.model_dump(exclude={"args"}), args=self.get_args()
        )

    def kwargs_for_new_context(self) -> BrowserNewContextArgs:
        return BrowserNewContextArgs(**self.model_dump(exclude={"args"}))

    def kwargs_for_launch(self) -> BrowserLaunchArgs:
        return BrowserLaunchArgs(
            **self.model_dump(exclude={"args"}), args=self.get_args()
        )

    def prepare_user_data_dir(self) -> None:
        if self.user_data_dir:
            self.user_data_dir = Path(self.user_data_dir).expanduser().resolve()
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            singleton_lock = self.user_data_dir / "SingletonLock"
            if singleton_lock.exists():
                try:
                    singleton_lock.unlink()
                except OSError:
                    logger.warning(
                        f"Could not remove SingletonLock at {singleton_lock}"
                    )

    def detect_display_configuration(self) -> None:
        if self.headless is None:
            self.headless = True
        if self.headless:
            self.window_size = None
            self.no_viewport = False
            # MODIFIED: Ensure assignment uses the PydanticCompatible type
            self.viewport = self.viewport or PydanticCompatibleViewportSize(
                width=1280, height=1024
            )
        else:
            # MODIFIED: Ensure assignment uses the PydanticCompatible type
            self.window_size = self.window_size or PydanticCompatibleViewportSize(
                width=1280, height=1024
            )
            self.no_viewport = True if self.no_viewport is None else self.no_viewport
            self.viewport = None


DEFAULT_BROWSER_PROFILE = BrowserProfile()

# --- The rest of your code (DOM, Agent Views, BrowserSession, Actions, Controller, MessageManager, Agent, main) remains unchanged ---
# ... (Make sure to include the rest of your original script from here)
# For brevity, the rest of the script is omitted but should be included in your file.


# --- Agent Views (Simplified) ---
class ActionResult(BaseModel):
    is_done: bool = False
    success: Optional[bool] = None
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False


@dataclass
class TabInfo:
    page_id: int
    url: str
    title: str


@dataclass
class BrowserStateSummary:  # Does not inherit from the new DOMState
    url: str
    title: str
    tabs: List[TabInfo]
    element_tree: DOMElementNode  # This is now from browspi.dom.views
    selector_map: SelectorMap  # This is now from browspi.dom.views
    screenshot: Optional[str] = None
    pixels_above: int = 0
    pixels_below: int = 0


class BrowserError(Exception):
    """Base class for all browser errors"""


class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed"""


# --- Browser Session (Simplified for prompt logic focus) ---
class BrowserSession(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, frozen=False)
    browser_profile: BrowserProfile = Field(default_factory=BrowserProfile)
    playwright: Optional[Any] = Field(default=None, exclude=True)
    browser: Optional[PlaywrightBrowser] = Field(default=None, exclude=True)
    browser_context: Optional[PlaywrightBrowserContext] = Field(
        default=None, exclude=True
    )
    agent_current_page: Optional[Page] = Field(default=None, exclude=True)
    initialized: bool = False
    _cached_browser_state_summary: Optional[BrowserStateSummary] = PrivateAttr(
        default=None
    )

    async def start(self) -> "BrowserSession":
        self.browser_profile.prepare_user_data_dir()
        self.browser_profile.detect_display_configuration()
        if not self.playwright:
            self.playwright = await playwright_async_playwright().start()  # type: ignore

        # --- MODIFICATION START for persistent context ---
        # Check if user_data_dir and executable_path are set in the profile,
        # implying persistent context mode is desired for this configuration.
        if self.browser_profile.user_data_dir and self.browser_profile.executable_path:
            logger.info(
                f"Attempting to launch persistent context with user_data_dir: {self.browser_profile.user_data_dir}"
            )

            # Get all kwargs for launch_persistent_context from the profile
            persistent_kwargs = (
                self.browser_profile.kwargs_for_launch_persistent_context().model_dump(
                    exclude_none=True
                )
            )

            # user_data_dir is the first positional argument for launch_persistent_context.
            # It should be present in persistent_kwargs if set in BrowserProfile.
            user_data_dir_to_launch = persistent_kwargs.pop("user_data_dir", None)
            if (
                not user_data_dir_to_launch
            ):  # Should ideally not happen if logic is correct
                raise ValueError(
                    "user_data_dir must be set in BrowserProfile for persistent context and was not found in launch kwargs."
                )

            # Ensure it's a Path object, though string should also work with Playwright
            user_data_dir_path = Path(user_data_dir_to_launch)

            self.browser_context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir_path,  # First argument is the user_data_dir
                **persistent_kwargs,  # Remaining arguments (executable_path, args, headless, etc.)
            )
            self.browser = (
                self.browser_context.browser
            )  # Get the browser instance from the context
        else:
            # Original behavior: launch a new browser instance and a new context
            logger.info(
                "Launching browser with a new context (not using a persistent user profile)."
            )
            if not self.browser:
                self.browser = await self.playwright.chromium.launch(
                    **self.browser_profile.kwargs_for_launch().model_dump(
                        exclude_none=True
                    )
                )
            if not self.browser_context:
                self.browser_context = await self.browser.new_context(
                    **self.browser_profile.kwargs_for_new_context().model_dump(
                        exclude_none=True
                    )
                )
        # --- MODIFICATION END ---

        if not self.browser_context.pages:  # type: ignore
            self.agent_current_page = await self.browser_context.new_page()  # type: ignore
        else:
            self.agent_current_page = self.browser_context.pages[0]  # type: ignore

        self.initialized = True
        logger.info(
            f"BrowserSession started. Using {'persistent context at ' + str(self.browser_profile.user_data_dir) if self.browser_profile.user_data_dir and self.browser_profile.executable_path else 'new context'}."
        )  # Enhanced logging
        return self

    async def stop(self):
        if not self.browser_profile.keep_alive:
            if self.browser_context:
                await self.browser_context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        self.initialized = False
        logger.info("BrowserSession stopped.")

    async def __aenter__(self) -> "BrowserSession":
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def get_current_page(self) -> Page:
        if (
            not self.initialized
            or not self.agent_current_page
            or self.agent_current_page.is_closed()
        ):
            if not self.browser_context or not self.browser_context.pages:
                self.agent_current_page = await self.browser_context.new_page()  # type: ignore
            else:
                self.agent_current_page = self.browser_context.pages[0]  # type: ignore
        return self.agent_current_page  # type: ignore

    # Inside class BrowserSession(BaseModel):

    async def get_state_summary(
        self, cache_clickable_elements_hashes: bool = True
    ) -> BrowserStateSummary:
        page = await self.get_current_page()
        url = page.url
        title = await page.title() or ""

        # Initialize DomService from your browspi package
        dom_service = DomService(page)  # This is now browspi.dom.service.DomService

        # These types are now from browspi.dom.views
        element_tree_result: Optional[DOMElementNode] = None
        selector_map_result: SelectorMap = {}

        try:
            # Call DomService to get the real DOM structure
            # highlight_elements and viewport_expansion are in self.browser_profile
            dom_state_from_service: DOMState = await dom_service.get_clickable_elements(
                highlight_elements=self.browser_profile.highlight_elements,
                viewport_expansion=self.browser_profile.viewport_expansion,
                # focus_element can be added if needed from profile
            )
            element_tree_result = dom_state_from_service.element_tree
            selector_map_result = dom_state_from_service.selector_map
            logger.debug(
                f"DomService returned {len(selector_map_result)} clickable elements for {url}."
            )

        except Exception as e:
            logger.error(
                f"Error getting clickable elements from DomService: {e}", exc_info=True
            )
            # Fallback to a minimal empty DOMElementNode
            element_tree_result = DOMElementNode(  # Ensure this matches browspi.dom.views.DOMElementNode constructor
                tag_name="body",
                xpath="/html/body",
                attributes={},
                children=[],
                is_visible=False,
                parent=None,
                # Add other mandatory fields for DOMElementNode if any, with default values
            )
            selector_map_result = {}

        # Screenshot logic (remains the same)
        screenshot_b64 = None
        try:
            screenshot_bytes = await page.screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"Could not take screenshot: {e}")

        # Pixels below logic (remains the same)
        pixels_below = 0
        try:
            pixels_below_eval = await page.evaluate(
                "document.body.scrollHeight - (window.innerHeight + window.scrollY)"
            )
            pixels_below = (
                int(pixels_below_eval)
                if isinstance(pixels_below_eval, (int, float))
                else 0
            )
        except Exception as e:
            logger.warning(f"Could not evaluate pixels_below: {e}")

        tabs_info = await self.get_tabs_info()  # Remains the same

        # Construct BrowserStateSummary with data from DomService
        self._cached_browser_state_summary = BrowserStateSummary(
            url=url,
            title=title,
            tabs=tabs_info,
            element_tree=element_tree_result,  # From DomService
            selector_map=selector_map_result,  # From DomService
            screenshot=screenshot_b64,
            pixels_above=0,  # You might want to implement this or get from DomService if available
            pixels_below=pixels_below,
        )

        # <<< ADD DETAILED LOGGING HERE TO DEBUG CLICKING ISSUES >>>
        # (Use the extensive logging block I provided in the previous response
        #  to log self._cached_browser_state_summary.element_tree.clickable_elements_to_string(...)
        #  and the contents of self._cached_browser_state_summary.selector_map)
        # Example to get you started (adapt from previous more detailed logging):
        logger.info(
            "--------------------------------------------------------------------"
        )
        logger.info(f"State for URL: {self._cached_browser_state_summary.url}")
        logger.info("LLM sees these clickable elements (first 20 lines):")
        try:
            attributes_to_include = self.browser_profile.include_attributes
            clickable_string = self._cached_browser_state_summary.element_tree.clickable_elements_to_string(
                attributes_to_include
            )
            for i, line in enumerate(clickable_string.split("\n")):
                if i < 20 and line.strip():
                    logger.info(f"  LLM_ELEMENT_VIEW: {line}")
                elif i == 20:
                    logger.info(
                        "  LLM_ELEMENT_VIEW: ... ( dalších prvků zkráceno )"
                    )  # ... (more elements truncated)
                    break
        except Exception as e_log_clickable:
            logger.error(
                f"Error generating clickable_elements_to_string for logging: {e_log_clickable}"
            )
        # Add detailed selector_map logging here too as per previous response.
        logger.info(
            "--------------------------------------------------------------------"
        )

        return self._cached_browser_state_summary

    async def take_screenshot(self, full_page: bool = False) -> str:
        page = await self.get_current_page()
        screenshot_bytes = await page.screenshot(full_page=full_page)
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def navigate(self, url: str):
        if not self._is_url_allowed(url):
            raise URLNotAllowedError(f"Navigation to {url} is not allowed.")
        page = await self.get_current_page()
        await page.goto(url, wait_until="domcontentloaded")

    def _is_url_allowed(self, url: str) -> bool:
        if not self.browser_profile.allowed_domains:
            return True
        parsed_url = urlparse(url)
        domain = parsed_url.hostname
        if not domain:
            return False  # Or handle as per your logic for invalid URLs
        return any(
            (
                domain.endswith(allowed_domain.lstrip("*."))
                if allowed_domain.startswith("*.")
                else domain == allowed_domain
            )
            for allowed_domain in self.browser_profile.allowed_domains
        )  # type: ignore

    async def get_tabs_info(self) -> List[TabInfo]:
        if not self.browser_context:
            return []
        tabs_info = []
        for i, page in enumerate(self.browser_context.pages):
            try:
                tabs_info.append(
                    TabInfo(page_id=i, url=page.url, title=await page.title())
                )
            except Exception:  # Handle pages that might be closed or inaccessible
                tabs_info.append(
                    TabInfo(
                        page_id=i, url="about:blank", title="Error retrieving title"
                    )
                )
        return tabs_info

    async def close_tab(self, page_id: Optional[int] = None):
        if not self.browser_context:
            return
        pages = self.browser_context.pages
        if not pages:
            return

        target_page = None
        if page_id is not None and 0 <= page_id < len(pages):
            target_page = pages[page_id]
        elif self.agent_current_page and not self.agent_current_page.is_closed():
            target_page = self.agent_current_page

        if target_page:
            await target_page.close()
            if self.agent_current_page == target_page:
                self.agent_current_page = None
            await self.get_current_page()  # Resets current page if needed


# --- Action Models (Simplified) ---
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
                return
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


# --- Controller and Registry (Simplified) ---
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
        # Simplified for brevity
        return f"{self.description}: {self.name}({self.param_model.__name__})"


class ActionRegistry(BaseModel):
    actions: Dict[str, RegisteredAction] = {}

    def get_prompt_description(self, page: Optional[Page] = None) -> str:
        # Simplified for brevity
        return "\n".join(
            action.prompt_description() for action in self.actions.values()
        )

    def create_action_model(
        self, include_actions: Optional[List[str]] = None, page: Optional[Page] = None
    ) -> Type[ActionModel]:
        fields = {}
        for name, action_info in self.actions.items():
            if include_actions and name not in include_actions:
                continue
            # Simplified filtering for this example
            if (
                page
                and action_info.domains
                and not any(
                    urlparse(page.url).hostname.endswith(d.lstrip("*."))
                    for d in action_info.domains
                )
            ):  # type: ignore
                continue
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
        if output_model:  # Simplified output model handling

            class CustomDoneAction(BaseModel):
                data: output_model
                success: bool  # type: ignore

            self.registry.actions["done"] = RegisteredAction(
                name="done",
                description="Completes the task with custom output",
                function=self._custom_done_action_func,
                param_model=CustomDoneAction,
            )

    async def _custom_done_action_func(
        self, params: BaseModel
    ):  # Actually CustomDoneAction
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
            if not actual_param_model:  # Auto-generate Pydantic model
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
                **kwargs,  # type: ignore
            )
            return func

        return decorator

    def _register_default_actions(self):
        # Simplified registration of default actions
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
            # Placeholder for click logic
            logger.info(f"Attempting to click element with index: {params.index}")
            # In a real scenario, you'd interact with Playwright here.
            # This requires DOM information from get_state_summary's selector_map.
            # For this prompt-focused version, we'll assume success or mock it.
            state = await browser_session.get_state_summary(
                cache_clickable_elements_hashes=False
            )
            if params.index in state.selector_map:
                element_to_click = state.selector_map[params.index]
                page = await browser_session.get_current_page()
                try:
                    # This is a simplified click, real version is more robust
                    await page.locator(f"xpath={element_to_click.xpath}").click(
                        timeout=5000
                    )  # type: ignore
                    return ActionResult(
                        extracted_content=f"Clicked element at index {params.index} ({element_to_click.tag_name})",
                        include_in_memory=True,
                    )
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
                    )  # type: ignore
                    return ActionResult(
                        extracted_content=f"Inputted '{params.text}' into element {params.index}",
                        include_in_memory=True,
                    )
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
        async def done(params: DoneAction):  # Removed browser_session as it's not used
            return ActionResult(
                is_done=True, success=params.success, extracted_content=params.text
            )

        @self.action("Open URL in a new tab", param_model=OpenTabAction)
        async def open_tab(params: OpenTabAction, browser_session: BrowserSession):
            if not browser_session.browser_context:
                await browser_session.start()  # Ensure context exists
            new_page = await browser_session.browser_context.new_page()  # type: ignore
            await new_page.goto(params.url, wait_until="domcontentloaded")
            browser_session.agent_current_page = (
                new_page  # Update current page reference
            )
            return ActionResult(
                extracted_content=f"Opened new tab with URL: {params.url}",
                include_in_memory=True,
            )

        @self.action("Close an existing tab by its ID", param_model=CloseTabAction)
        async def close_tab(params: CloseTabAction, browser_session: BrowserSession):
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
            # Simplified: Get all text content. Real version uses LLM for targeted extraction.
            text_content = (
                await page.content()
            )  # Or page.inner_text('body') for less HTML
            # In a real scenario, you would use page_extraction_llm with text_content and params.goal
            # For now, just returning a snippet or a message.
            summary = f"Content related to '{params.goal}'. (Full content length: {len(text_content)})"
            if page_extraction_llm:
                # Simplified LLM call, actual prompt would be more sophisticated
                try:
                    llm_response = await page_extraction_llm.ainvoke(
                        f"Extract information about '{params.goal}' from the following text:\n{text_content[:4000]}"
                    )  # Limit input size
                    summary = llm_response.content  # type: ignore
                except Exception as e:
                    logger.error(f"LLM extraction failed: {e}")
                    summary = f"Could not extract with LLM. Raw content for '{params.goal}' first 200 chars: {text_content[:200]}"

            return ActionResult(extracted_content=summary, include_in_memory=True)

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
        async def wait(params: WaitAction):  # Removed browser_session as it's not used
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
        action_name = ""
        # The model_dump(exclude_unset=True) ensures we only get explicitly set fields.
        # This is important as ActionModel can have many optional fields for different actions.
        dumped_action = action.model_dump(exclude_unset=True)
        if not dumped_action:
            return ActionResult(error="No action specified in the model.")

        action_name = list(dumped_action.keys())[0]
        params_obj = dumped_action[action_name]

        if action_name not in self.registry.actions:
            return ActionResult(error=f"Action '{action_name}' not found.")

        registered_action = self.registry.actions[action_name]

        # Prepare kwargs for the action function
        action_kwargs: Dict[str, Any] = {}
        sig = inspect.signature(registered_action.function)

        # Pass BrowserSession if the action expects it
        if "browser_session" in sig.parameters:
            action_kwargs["browser_session"] = browser_session
        if "page_extraction_llm" in sig.parameters:
            action_kwargs["page_extraction_llm"] = page_extraction_llm
        # Add other context-specific params as needed

        try:
            # If params_obj is already a Pydantic model instance (because it was nested)
            if isinstance(params_obj, BaseModel):
                validated_params = params_obj
            # If params_obj is a dictionary, validate it using the action's param_model
            elif isinstance(params_obj, dict):
                validated_params = registered_action.param_model(**params_obj)
            else:
                # If params_obj is None (for actions with no parameters), create an empty instance
                # This assumes NoParamsAction or similar for actions without params
                validated_params = registered_action.param_model()

            # Execute the action
            # If the action function expects its parameters as a single Pydantic model:
            if (
                len(sig.parameters) > 0
                and list(sig.parameters.values())[0].name != "self"
            ):
                first_param_name = list(sig.parameters.keys())[0]
                first_param_annotation = sig.parameters[first_param_name].annotation
                if inspect.isclass(first_param_annotation) and issubclass(
                    first_param_annotation, BaseModel
                ):
                    result = await registered_action.function(
                        validated_params, **action_kwargs
                    )
                else:  # Action expects individual kwargs
                    result = await registered_action.function(
                        **validated_params.model_dump(exclude_none=True),
                        **action_kwargs,
                    )  # type: ignore
            else:  # Action expects individual kwargs or no args beyond context
                result = await registered_action.function(
                    **validated_params.model_dump(exclude_none=True), **action_kwargs
                )  # type: ignore

            if isinstance(result, ActionResult):
                return result
            if isinstance(result, str):
                return ActionResult(extracted_content=result)
            return ActionResult()  # Default if action returns None or unhandled type
        except PlaywrightTimeoutError:
            logger.error(f"Timeout error during action: {action_name}")
            return ActionResult(error=f"Action '{action_name}' timed out.")
        except PlaywrightError as e:
            logger.error(f"Playwright error during action {action_name}: {e}")
            return ActionResult(error=f"Browser error during '{action_name}': {str(e)}")
        except ValidationError as e:
            logger.error(f"Parameter validation error for action {action_name}: {e}")
            return ActionResult(
                error=f"Invalid parameters for '{action_name}': {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {e}", exc_info=True)
            return ActionResult(
                error=f"Unexpected error during '{action_name}': {str(e)}"
            )


# --- Message Manager (Simplified) ---
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

    def remove_last_state_message(self):  # Removes the HumanMessage with browser state
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
    estimated_characters_per_token: int = 3  # Rough estimate
    image_tokens: int = 800  # OpenAI's cost for a low-detail image tile
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
        ]
    )
    message_context: Optional[str] = None
    sensitive_data: Optional[Dict[str, str]] = None
    available_file_paths: Optional[List[str]] = None


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
        # Simplified token counting
        content_str = ""
        if isinstance(message.content, str):
            content_str = message.content
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_str += item["text"]
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    content_str += " [IMAGE] "  # Placeholder for image tokens
        if hasattr(message, "tool_calls") and message.tool_calls:  # type: ignore
            content_str += str(message.tool_calls)  # type: ignore

        # Basic character-based estimation + image token cost
        char_tokens = len(content_str) // self.settings.estimated_characters_per_token
        image_count = content_str.count("[IMAGE]")
        return char_tokens + (image_count * self.settings.image_tokens)

    def _add_message_with_tokens(
        self,
        message: BaseMessage,
        position: Optional[int] = None,
        message_type: Optional[str] = None,
    ):
        # Simplified sensitive data filtering for this example
        if self.settings.sensitive_data and isinstance(message.content, str):
            temp_content = message.content
            for placeholder, real_value in self.settings.sensitive_data.items():
                if real_value:  # only replace if real_value is not empty
                    temp_content = temp_content.replace(
                        real_value, f"<secret>{placeholder}</secret>"
                    )
            message.content = temp_content
        elif self.settings.sensitive_data and isinstance(message.content, list):
            new_content_list = []
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    temp_text = item["text"]
                    for placeholder, real_value in self.settings.sensitive_data.items():
                        if real_value:
                            temp_text = temp_text.replace(
                                real_value, f"<secret>{placeholder}</secret>"
                            )
                    new_content_list.append({"type": "text", "text": temp_text})
                else:
                    new_content_list.append(item)
            message.content = new_content_list

        token_count = self._count_tokens(message)
        self.state.history.add_message(
            message,
            MessageMetadata(tokens=token_count, message_type=message_type),
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
            info = f"Sensitive data placeholders: {list(self.settings.sensitive_data.keys())}. Use them like <secret>placeholder_name</secret>"
            self._add_message_with_tokens(
                HumanMessage(content=info), message_type="init"
            )

        # Simplified example output for OpenAI (tool_calls format)
        self._add_message_with_tokens(
            HumanMessage(content="Example output format:"), message_type="init"
        )
        # Inside MessageManager._init_messages method
        arguments_dict = {
            "current_state": {
                "evaluation_previous_goal": "Success - I did X.",
                "memory": "I have done X and Y. Current step 3/15.",
                "next_goal": "Now I will do Z.",
            },
            "action": [{"go_to_url": {"url": "https://example.com"}}],
        }
        example_tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": str(self.state.tool_id),
                    "name": "AgentOutput",  # 'name' is now top-level
                    "args": arguments_dict,  # 'args' is now top-level and a dictionary
                }
            ],
        )
        self._add_message_with_tokens(example_tool_call, message_type="init")
        self.add_tool_message(
            content="Browser started", message_type="init"
        )  # Tool response for the example

        self._add_message_with_tokens(
            HumanMessage(content="[Your task history memory starts here]"),
            message_type="init",
        )
        if self.settings.available_file_paths:
            self._add_message_with_tokens(
                HumanMessage(
                    content=f"Available file paths: {self.settings.available_file_paths}"
                ),
                message_type="init",
            )

    def add_new_task(self, new_task: str):
        self.task = new_task  # Update the main task
        self._add_message_with_tokens(
            HumanMessage(
                content=f'Your new ultimate task is: """{new_task}""". Consider previous context.'
            )
        )

    def add_state_message(
        self,
        browser_state_summary: BrowserStateSummary,
        result: Optional[List[ActionResult]] = None,
        step_info: Optional[Any] = None,
        use_vision: bool = True,
    ):
        state_text = f"\n[Current state starts here]\nCurrent URL: {browser_state_summary.url}\nOpen Tabs: {[(tab.page_id, tab.title) for tab in browser_state_summary.tabs]}\n"
        state_text += f"Visible elements:\n{browser_state_summary.element_tree.clickable_elements_to_string(self.settings.include_attributes)}\n"
        if step_info:
            state_text += f"Step: {step_info.step_number + 1}/{step_info.max_steps}\n"
        state_text += f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"

        if result:
            for i, res_item in enumerate(result):
                if res_item.extracted_content:
                    state_text += (
                        f"Action Result {i + 1}: {res_item.extracted_content}\n"
                    )
                if res_item.error:
                    state_text += f"Action Error {i + 1}: {res_item.error.splitlines()[-1]}\n"  # Only last line of error

        content_parts: list[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": state_text}
        ]
        if use_vision and browser_state_summary.screenshot:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{browser_state_summary.screenshot}"
                    },
                }
            )
        self._add_message_with_tokens(HumanMessage(content=content_parts))  # type: ignore

    def add_model_output(self, model_output: "AgentOutput"):
        # For OpenAI, the model output (AgentOutput) needs to be formatted as a tool call
        # The 'AgentOutput' tool should be defined in the LLM's tool configuration
        tool_calls = [
            {
                "id": str(self.state.tool_id),
                "name": "AgentOutput",  # 'name' is now top-level
                "args": model_output.model_dump(
                    exclude_unset=True, exclude_none=True
                ),  # 'args' is top-level and a dict
            }
        ]
        ai_message = AIMessage(content="", tool_calls=tool_calls)
        self._add_message_with_tokens(ai_message)
        self.add_tool_message(content="Action processed.")  # Placeholder tool response

    def add_tool_message(self, content: str, message_type: Optional[str] = None):
        # For OpenAI, ToolMessage needs a tool_call_id
        tool_message = ToolMessage(
            content=content, tool_call_id=str(self.state.tool_id)
        )
        self.state.tool_id += 1  # Increment for the next tool call
        self._add_message_with_tokens(tool_message, message_type=message_type)

    def get_messages(self) -> List[BaseMessage]:
        return [m.message for m in self.state.history.messages]

    def _ensure_token_limit(self):
        # Simplified token management: if over limit, remove oldest non-init message
        while (
            self.state.history.current_tokens > self.settings.max_input_tokens
            and len(self.state.history.messages) > 1
        ):
            for i, msg_container in enumerate(self.state.history.messages):
                if msg_container.metadata.message_type != "init":
                    removed_msg = self.state.history.messages.pop(i)
                    self.state.history.current_tokens -= removed_msg.metadata.tokens
                    logger.info(
                        f"Removed message to stay within token limit. Current tokens: {self.state.history.current_tokens}"
                    )
                    break
            else:  # No non-init message found to remove, break loop
                break

    def cut_messages(self):  # Placeholder
        pass


# --- Agent Prompts (Simplified) ---
class SystemPrompt:
    def __init__(
        self,
        action_description: str,
        max_actions_per_step: int = 10,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
    ):
        prompt = ""
        if override_system_message:
            prompt = override_system_message
        else:
            # Simplified default prompt for OpenAI
            prompt = f"""You are an AI agent that interacts with a web browser based on user tasks.
Your goal is to perform actions to achieve the task.
You will be given the current state of the browser (URL, visible elements, screenshot).
Respond with a JSON object using the 'AgentOutput' tool.
The JSON should contain:
1. `current_state`: Your assessment of the previous action, current progress, and next immediate goal.
    - `evaluation_previous_goal`: "Success", "Failed", or "Unknown", with a brief explanation.
    - `memory`: What you've done and need to remember (e.g., "Searched X, clicked Y. 2/5 steps done.").
    - `next_goal`: Your immediate next objective (e.g., "Click the 'Login' button.").
2. `action`: A list of one or more actions to perform. Max {max_actions_per_step} actions.
    Available actions:
    {action_description}

Example of a single action:
"action": [{{"go_to_url": {{"url": "https://example.com"}}}}]

Example of multiple actions (fill form and click):
"action": [
    {{"input_text": {{"index": 1, "text": "my_username"}}}},
    {{"input_text": {{"index": 2, "text": "my_password"}}}},
    {{"click_element_by_index": {{"index": 3}}}}
]

If the task is completed, use the "done" action with `success: true` and a summary in `text`.
If you reach the max steps and the task isn't fully done, use "done" with `success: false`.

Tips:
When submitting a search query you have typed, prioritize clicking a button with explicit text like 'Google Search' or a button with type='submit'. Avoid clicking auxiliary search options like 'Search by image' or 'Search by voice' for the primary search action unless the task specifically requires it.
"""
        if extend_system_message:
            prompt += f"\n{extend_system_message}"
        self.system_message = SystemMessage(content=prompt)

    def get_system_message(self) -> SystemMessage:
        return self.system_message


class AgentMessagePrompt:  # Retained from original structure, used by MessageManager
    def __init__(
        self,
        browser_state_summary: BrowserStateSummary,
        result: Optional[List[ActionResult]] = None,
        include_attributes: Optional[List[str]] = None,
        step_info: Optional[Any] = None,
    ):
        self.state = browser_state_summary
        self.result = result
        self.include_attributes = include_attributes or []
        self.step_info = step_info

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        # Simplified for brevity, actual implementation is more detailed.
        elements_text = self.state.element_tree.clickable_elements_to_string(
            include_attributes=self.include_attributes
        )
        state_description = f"\n[Current state starts here]\nCurrent URL: {self.state.url}\nTabs: {self.state.tabs}\nInteractive elements:\n{elements_text}\n"
        if self.step_info:
            state_description += (
                f"Step: {self.step_info.step_number + 1}/{self.step_info.max_steps}\n"
            )
        state_description += (
            f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        )

        if self.result:
            for i, res_item in enumerate(self.result):
                if res_item.extracted_content:
                    state_description += (
                        f"Action Result {i + 1}: {res_item.extracted_content}\n"
                    )
                if res_item.error:
                    state_description += (
                        f"Action Error {i + 1}: {res_item.error.splitlines()[-1]}\n"
                    )

        content_parts: list[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": state_description}
        ]
        if use_vision and self.state.screenshot:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.state.screenshot}"
                    },
                }
            )
        return HumanMessage(content=content_parts)  # type: ignore


class AgentBrain(BaseModel):
    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_state: AgentBrain
    action: List[ActionModel] = Field(
        ..., min_length=1
    )  # This will use the dynamically created ActionModel

    @staticmethod
    def type_with_custom_actions(
        custom_actions_model: Type[ActionModel],
    ) -> Type["AgentOutput"]:
        # Ensure List generic alias is correctly used for Pydantic
        return create_model(
            "DynamicAgentOutput",
            __base__=AgentOutput,  # Base class is AgentOutput itself
            action=(List[custom_actions_model], Field(..., min_length=1)),  # type: ignore
            __module__=AgentOutput.__module__,  # Keep the same module for clarity
        )


class AgentSettings(BaseModel):
    use_vision: bool = True
    save_conversation_path: Optional[str] = None
    max_failures: int = 3
    retry_delay: int = 10
    override_system_message: Optional[str] = None
    extend_system_message: Optional[str] = None
    max_input_tokens: int = 128000  # Default for gpt-4o
    validate_output: bool = False  # Not fully implemented in this replica
    message_context: Optional[str] = None
    generate_gif: Union[bool, str] = False  # Not implemented in this replica
    available_file_paths: Optional[List[str]] = None
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
        ]
    )
    max_actions_per_step: int = 10
    tool_calling_method: Optional[
        Literal["function_calling", "json_mode", "raw", "auto", "tools"]
    ] = "auto"
    page_extraction_llm: Optional[BaseChatModel] = (
        None  # LLM for extraction, if different
    )
    planner_llm: Optional[BaseChatModel] = None  # Not implemented in this replica
    planner_interval: int = 1
    is_planner_reasoning: bool = False
    save_playwright_script_path: Optional[str] = None  # Not implemented
    extend_planner_system_message: Optional[str] = None


class AgentState(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: Optional[List[ActionResult]] = None
    history: "AgentHistoryList" = Field(
        default_factory=lambda: AgentHistoryList(history=[])
    )  # Forward ref
    message_manager_state: MessageManagerState = Field(
        default_factory=MessageManagerState
    )
    paused: bool = False
    stopped: bool = False


class AgentHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_output: Optional[AgentOutput] = None  # This should be the dynamic AgentOutput
    result: List[ActionResult]
    # Simplified state for history, real one is BrowserStateHistory
    state: Dict[str, Any]  # Stores simplified URL, title, screenshot
    metadata: Optional[Dict[str, Any]] = None  # Simplified metadata


class AgentHistoryList(BaseModel):
    history: List[AgentHistory] = Field(default_factory=list)

    def is_done(self) -> bool:
        return bool(
            self.history
            and self.history[-1].result
            and self.history[-1].result[-1].is_done
        )

    def is_successful(self) -> Optional[bool]:
        if self.is_done():
            return self.history[-1].result[-1].success
        return None

    def final_result(self) -> Optional[str]:
        if self.is_done():
            return self.history[-1].result[-1].extracted_content
        return None


# AgentState needs AgentHistoryList definition
AgentState.model_rebuild()


# --- Agent Service ---
class Agent(Generic[Context]):
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: Optional[BrowserSession] = None,
        controller: Optional[Controller[Context]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        # Agent settings (subset of original for focus)
        use_vision: bool = True,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
        max_input_tokens: int = 120000,  # Slightly less than 128k for safety
        max_actions_per_step: int = 5,  # Reduced for simplicity
        page_extraction_llm: Optional[BaseChatModel] = None,
        injected_agent_state: Optional[AgentState] = None,
        context: Optional[Context] = None,
        source: Optional[str] = None,  # For telemetry, can be ignored here
    ):
        self.task = task
        self.llm = llm
        self.controller = controller or Controller()
        self.sensitive_data = sensitive_data
        self.version = "replica-0.1"  # Mock version
        self.source = source or "replica"

        self.settings = AgentSettings(
            use_vision=use_vision,
            override_system_message=override_system_message,
            extend_system_message=extend_system_message,
            max_input_tokens=max_input_tokens,
            max_actions_per_step=max_actions_per_step,
            page_extraction_llm=page_extraction_llm
            or llm,  # Default to main LLM if not provided
        )

        self.state = injected_agent_state or AgentState()
        self.browser_session = browser_session or BrowserSession(
            browser_profile=DEFAULT_BROWSER_PROFILE
        )
        self.context = context

        # --- Setup ActionModel and AgentOutput based on controller ---
        self.ActionModelType = (
            self.controller.registry.create_action_model()
        )  # Store the type
        self.AgentOutputType = AgentOutput.type_with_custom_actions(
            self.ActionModelType
        )  # Store the type
        # ---

        self.initial_actions = (
            self._convert_initial_actions(initial_actions) if initial_actions else None
        )

        self.tool_calling_method = "tools"
        if not hasattr(self.llm, "bind_tools"):
            logger.warning(
                f"LLM {self.llm.__class__.__name__} may not support bind_tools. Tool calling might fail."
            )

        self._message_manager = MessageManager(
            task=task,
            system_message=SystemPrompt(
                action_description=self.controller.registry.get_prompt_description(),
                max_actions_per_step=self.settings.max_actions_per_step,
                override_system_message=self.settings.override_system_message,
                extend_system_message=self.settings.extend_system_message,
            ).get_system_message(),
            settings=MessageManagerSettings(
                max_input_tokens=self.settings.max_input_tokens,
                include_attributes=self.settings.include_attributes,
                message_context=self.settings.message_context,
                sensitive_data=self.sensitive_data,
                available_file_paths=self.settings.available_file_paths,
            ),
            state=self.state.message_manager_state,
        )
        self._external_pause_event = asyncio.Event()
        self._external_pause_event.set()

    def _convert_initial_actions(
        self, actions: List[Dict[str, Dict[str, Any]]]
    ) -> List[ActionModel]:  # Return list of ActionModel instances
        converted_actions = []
        for action_dict in actions:
            try:
                action_instance = self.ActionModelType(
                    **action_dict
                )  # Use the stored type
                converted_actions.append(action_instance)
            except ValidationError as e:
                logger.error(f"Failed to validate initial action {action_dict}: {e}")
        return converted_actions

    async def get_next_action(
        self, input_messages: List[BaseMessage]
    ) -> AgentOutput:  # Return instance of AgentOutput
        try:
            llm_with_tool = self.llm.bind_tools(
                tools=[self.AgentOutputType], tool_choice=self.AgentOutputType.__name__
            )  # type: ignore
        except Exception as e:
            logger.error(
                f"Failed to bind tools to LLM. Ensure your LLM supports tool calling. Error: {e}"
            )
            llm_with_tool = self.llm

        raw_response = await llm_with_tool.ainvoke(input_messages)

        if not hasattr(raw_response, "tool_calls") or not raw_response.tool_calls:  # type: ignore
            logger.warning(
                "LLM did not use tool_calls. Attempting to parse content as JSON."
            )
            if isinstance(raw_response.content, str):
                try:
                    content_str = raw_response.content
                    if content_str.startswith("```json"):
                        content_str = content_str[7:]
                    if content_str.endswith("```"):
                        content_str = content_str[:-3]
                    content_str = content_str.strip()
                    parsed_json = json.loads(content_str)
                    model_output = self.AgentOutputType(
                        **parsed_json
                    )  # Use the stored type
                    log_response(model_output)
                    return model_output
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(
                        f"Failed to parse LLM content as AgentOutput JSON: {e}. Content: {raw_response.content}"
                    )
                    raise ValueError(
                        f"Could not parse LLM response: {raw_response.content}"
                    ) from e
            else:
                logger.error(
                    f"LLM response content is not a string and no tool_calls found. Response: {raw_response}"
                )
                raise ValueError("LLM response is in an unexpected format.")

        tool_call = raw_response.tool_calls[0]  # type: ignore
        if (
            tool_call["name"].lower() != self.AgentOutputType.__name__.lower()
        ):  # Compare with stored type name
            logger.error(f"LLM called unexpected tool: {tool_call['name']}")
            raise ValueError(f"LLM called unexpected tool: {tool_call['name']}")

        try:
            if isinstance(tool_call["args"], str):
                model_output = self.AgentOutputType.model_validate_json(
                    tool_call["args"]
                )  # Use stored type
            elif isinstance(tool_call["args"], dict):
                model_output = self.AgentOutputType(
                    **tool_call["args"]
                )  # Use stored type
            else:
                raise ValueError(
                    f"Tool arguments are not a valid string or dict: {tool_call['args']}"
                )
        except ValidationError as e:
            logger.error(
                f"Failed to validate AgentOutput from tool call arguments: {e}. Args: {tool_call['args']}"
            )
            if isinstance(tool_call["args"], str):
                try:
                    args_dict = json.loads(tool_call["args"])
                    model_output = self.AgentOutputType(**args_dict)  # Use stored type
                except (json.JSONDecodeError, ValidationError) as e2:
                    logger.error(f"Retry parsing AgentOutput failed: {e2}")
                    raise ValueError(
                        f"Could not parse tool arguments for AgentOutput: {tool_call['args']}"
                    ) from e2
            else:
                raise ValueError(
                    f"Could not parse tool arguments for AgentOutput: {tool_call['args']}"
                ) from e

        log_response(model_output)
        return model_output

    async def multi_act(
        self, actions: List[ActionModel]
    ) -> List[ActionResult]:  # Parameter is List[ActionModel instance]
        results = []
        for i, action_model_instance in enumerate(actions):
            if not isinstance(
                action_model_instance, self.ActionModelType
            ):  # Check against stored type
                logger.error(
                    f"Action {i} is not an instance of the expected ActionModel. Got: {type(action_model_instance)}"
                )
                if isinstance(action_model_instance, dict):
                    try:
                        action_model_instance = self.ActionModelType(
                            **action_model_instance
                        )  # Use stored type
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
                action=action_model_instance,
                browser_session=self.browser_session,
                page_extraction_llm=page_extraction_llm,
                sensitive_data=self.sensitive_data,
                available_file_paths=self.settings.available_file_paths,
                context=self.context,
            )
            results.append(result)
            if result.is_done or result.error:
                break
            await asyncio.sleep(
                self.browser_session.browser_profile.wait_between_actions
            )
        return results

    async def _handle_step_error(self, error: Exception) -> List[ActionResult]:
        # Simplified error handling
        error_msg = str(error)
        logger.error(f"Step failed: {error_msg}", exc_info=True)
        self.state.consecutive_failures += 1
        return [ActionResult(error=error_msg, include_in_memory=True)]

    async def step(self, step_info: Optional[Any] = None):
        logger.info(f"--- Step {self.state.n_steps} ---")
        browser_state_summary = None
        model_output = None  # Should be instance of self.AgentOutputType
        result: List[ActionResult] = []

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
            model_output = await self.get_next_action(input_messages)

            self._message_manager.state.history.remove_last_state_message()
            self._message_manager.add_model_output(model_output)

            result = await self.multi_act(model_output.action)
            self.state.last_result = result
            self.state.consecutive_failures = 0

        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result
        finally:
            if browser_state_summary and model_output:
                self.state.history.history.append(
                    AgentHistory(
                        model_output=model_output,  # This is an instance of self.AgentOutputType
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
            self.state.last_result = await self.multi_act(self.initial_actions)

        for step_num in range(max_steps):
            if self.state.stopped:
                logger.info("Agent stopped.")
                break
            if self.state.paused:
                logger.info("Agent paused. Waiting for resume...")
                await self._external_pause_event.wait()
                logger.info("Agent resumed.")

            if self.state.consecutive_failures >= self.settings.max_failures:
                logger.error(
                    f"Stopping due to {self.settings.max_failures} consecutive failures."
                )
                break

            if on_step_start:
                await on_step_start(self)  # type: ignore

            @dataclass
            class StepInfo:
                step_number: int
                max_steps: int

            await self.step(
                step_info=StepInfo(step_number=step_num, max_steps=max_steps)
            )

            if on_step_end:
                await on_step_end(self)  # type: ignore

            if self.state.history.is_done():
                logger.info("✅ Task marked as done.")
                break
        else:
            logger.info(f"Max steps ({max_steps}) reached.")
            if not self.state.history.is_done():
                done_action_model_type = self.controller.registry.create_action_model(
                    include_actions=["done"]
                )
                # Create an instance of the specific ActionModel that can hold a DoneAction
                # The action field in AgentOutput expects a list of ActionModelType instances.
                # So, we create an instance of ActionModelType with the 'done' field set.
                final_action_instance = done_action_model_type(
                    done=DoneAction(
                        text="Max steps reached, task may be incomplete.", success=False
                    )
                )

                # Create an AgentBrain instance for the current_state
                final_agent_brain = AgentBrain(
                    evaluation_previous_goal="Max steps reached",
                    memory="N/A",
                    next_goal="N/A",
                )

                # Create an AgentOutput instance using the stored AgentOutputType
                # This ensures the 'action' field has the correct List[ActionModelType]
                final_model_output = self.AgentOutputType(
                    current_state=final_agent_brain,
                    action=[
                        final_action_instance
                    ],  # Pass the correctly typed action instance
                )

                self.state.history.history.append(
                    AgentHistory(
                        model_output=final_model_output,
                        result=[
                            ActionResult(
                                is_done=True,
                                success=False,
                                extracted_content="Max steps reached.",
                            )
                        ],
                        state={"url": "N/A", "title": "N/A"},
                        metadata={"step": self.state.n_steps},
                    )
                )

        logger.info(f"Agent run finished. Total steps: {self.state.n_steps - 1}")
        if self.settings.save_conversation_path:
            try:
                with open(
                    f"{self.settings.save_conversation_path}_final.json", "w"
                ) as f:
                    json.dump(self.state.history.model_dump(), f, indent=2)  # type: ignore
                logger.info(
                    f"Conversation history saved to {self.settings.save_conversation_path}_final.json"
                )
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

        return self.state.history

    async def close(self):
        await self.browser_session.stop()

    def pause(self):
        self.state.paused = True
        self._external_pause_event.clear()
        logger.info("Agent paused.")

    def resume(self):
        self.state.paused = False
        self._external_pause_event.set()
        logger.info("Agent resumed.")

    def stop(self):
        self.state.stopped = True
        logger.info("Agent stop requested.")


def log_response(
    response: AgentOutput,
) -> None:  # Parameter type is the base AgentOutput
    """Utility function to log the model's response."""
    emoji = "🤷"
    if response.current_state and response.current_state.evaluation_previous_goal:
        if "Success" in response.current_state.evaluation_previous_goal:
            emoji = "👍"
        elif "Failed" in response.current_state.evaluation_previous_goal:
            emoji = "⚠"
        logger.info(f"{emoji} Eval: {response.current_state.evaluation_previous_goal}")
        logger.info(f"🧠 Memory: {response.current_state.memory}")
        logger.info(f"🎯 Next goal: {response.current_state.next_goal}")

    if response.action:  # response.action will be List[ActionModelType instance]
        for i, action_model_instance in enumerate(response.action):
            action_dict = action_model_instance.model_dump(exclude_unset=True)
            logger.info(
                f"🛠️ Action {i + 1}/{len(response.action)}: {json.dumps(action_dict)}"
            )
    else:
        logger.warning("No action provided in AgentOutput.")


# --- Main Execution (Example) ---
async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY not found. Please set it in your environment or .env file."
        )
        return

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
    )

    task = "Go to Google.com, search for 'Python programming', and then click on the official Python website link if visible."

    # <<< START OF MODIFICATIONS FOR CHROME PROFILE >>>
    # User's provided paths and arguments
    persistent_profile_path = (
        "C:\\Users\\MSI\\AppData\\Local\\Google\\Chrome\\User Data\\Default"
    )
    chrome_exe_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    other_browser_args = ["--start-maximized"]

    browser_profile = BrowserProfile(
        user_data_dir=persistent_profile_path,  # Use the user's Chrome profile path
        executable_path=chrome_exe_path,  # Use the user's Chrome executable
        args=other_browser_args,  # Add custom browser arguments
        headless=False,  # Run in non-headless mode to see the browser
        # and for persistent profile to load correctly.
        # channel="chromium" is the default in BrowserProfile and should work with executable_path.
    )
    # <<< END OF MODIFICATIONS FOR CHROME PROFILE >>>

    browser_session = BrowserSession(browser_profile=browser_profile)

    agent = Agent(
        task=task,
        llm=llm,
        browser_session=browser_session,
        # Ensure other necessary Agent parameters are passed if you use them
    )

    try:
        print(f"Starting agent with task: {task}")
        history = await agent.run(max_steps=5)
        print("\n--- Agent Run History ---")
        if history.history:
            for i, item in enumerate(history.history):
                print(f"\n--- History Step {i + 1} ---")
                if (
                    item.model_output
                ):  # item.model_output is an instance of AgentOutputType
                    # Ensure AgentBrain attributes are accessed correctly if they exist
                    current_state = item.model_output.current_state
                    if current_state:
                        print(
                            f"  LLM Thought: {current_state.evaluation_previous_goal} -> {current_state.next_goal}"
                        )
                    for action_item in (
                        item.model_output.action
                    ):  # action_item is an instance of ActionModelType
                        print(
                            f"  LLM Action: {action_item.model_dump_json(exclude_unset=True)}"
                        )
                if item.result:
                    for res_item in item.result:
                        if res_item.extracted_content:
                            print(
                                f"  Action Result: {res_item.extracted_content[:200]}..."
                            )
                        if res_item.error:
                            print(f"  Action Error: {res_item.error}")
                        if res_item.is_done:
                            print(f"  Task Done: Success={res_item.success}")
                if item.state and item.state.get("url"):
                    print(
                        f"  Browser State: URL={item.state['url']}, Title={item.state.get('title')}"
                    )

        final_res = history.final_result()
        if final_res:
            print(f"\nFinal Result from Agent: {final_res}")
        else:
            print("\nAgent did not produce a final result or was not 'done'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error("Main execution error", exc_info=True)
    finally:
        print("Closing browser session...")
        await agent.close()


def __main__():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
