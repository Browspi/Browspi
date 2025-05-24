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

from browspi.services.clickable_element_processor.service import (
    ClickableElementProcessor,
)

# Assuming browspi.services.dom.service and browspi.services.views are available in your project
# If not, these would need to be created or adjusted.
# For this example, I'll mock them if they cause import errors later.
from browspi.services.dom.service import DomService
from browspi.services.views import DOMElementNode, SelectorMap

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
@dataclass
class CachedClickableElementHashes:
    url: str
    hashes: set[str]


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
    proxy: Optional[PydanticCompatibleProxySettings] = None
    user_agent: Optional[str] = None
    viewport: Optional[PydanticCompatibleViewportSize] = Field(default=None)
    no_viewport: Optional[bool] = None
    locale: Optional[str] = None
    geolocation: Optional[PydanticCompatibleGeolocation] = None
    timezone_id: Optional[str] = None
    color_scheme: ColorScheme = ColorScheme.LIGHT
    extra_http_headers: Dict[str, str] = Field(default_factory=dict)
    offline: bool = False
    http_credentials: Optional[PydanticCompatibleHttpCredentials] = None
    ignore_https_errors: bool = False
    java_script_enabled: bool = True
    bypass_csp: bool = False
    service_workers: Literal["allow", "block"] = "allow"
    client_certificates: List[PydanticCompatibleClientCertificate] = Field(
        default_factory=list
    )
    record_har_path: Optional[Union[str, Path]] = None
    record_video_dir: Optional[Union[str, Path]] = None
    record_video_size: Optional[PydanticCompatibleViewportSize] = None
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
    proxy: Optional[PydanticCompatibleProxySettings] = None
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


class BrowserNewContextArgs(BrowserContextArgs):
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=False,
        revalidate_instances="always",
        populate_by_name=True,
    )
    storage_state: Optional[Union[str, Path, Dict[str, Any]]] = None


class BrowserLaunchPersistentContextArgs(BrowserLaunchArgs, BrowserContextArgs):
    model_config = ConfigDict(
        extra="ignore", validate_assignment=False, revalidate_instances="always"
    )
    user_data_dir: Optional[Union[str, Path]] = Path("").expanduser()


class BrowserProfile(
    BrowserLaunchPersistentContextArgs, BrowserLaunchArgs, BrowserNewContextArgs
):
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
    window_size: Optional[PydanticCompatibleViewportSize] = Field(default=None)
    minimum_wait_page_load_time: float = Field(default=0.25)
    wait_for_network_idle_page_load_time: float = Field(default=0.5)
    maximum_wait_page_load_time: float = Field(default=5.0)
    wait_between_actions: float = Field(default=0.5)
    include_dynamic_attributes: bool = Field(default=True)
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
    highlight_elements: bool = Field(default=False)
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
            self.viewport = self.viewport or PydanticCompatibleViewportSize(
                width=1280, height=1024
            )
        else:
            self.window_size = self.window_size or PydanticCompatibleViewportSize(
                width=1280, height=1024
            )
            self.no_viewport = True if self.no_viewport is None else self.no_viewport
            self.viewport = None


DEFAULT_BROWSER_PROFILE = BrowserProfile()


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
class BrowserStateSummary:
    url: str
    title: str
    tabs: List[TabInfo]
    element_tree: DOMElementNode
    selector_map: SelectorMap
    screenshot: Optional[str] = None
    pixels_above: int = 0
    pixels_below: int = 0


class BrowserError(Exception):
    """Base class for all browser errors"""


class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed"""


def _log_pretty_url(s: str, max_len: int | None = 22) -> str:
    """Truncate/pretty-print a URL with a maximum length, removing the protocol and www. prefix"""
    s = s.replace("https://", "").replace("http://", "").replace("www.", "")
    if max_len is not None and len(s) > max_len:
        return s[:max_len] + "…"
    return s


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
    _cached_clickable_element_hashes: CachedClickableElementHashes | None = PrivateAttr(
        default=None
    )

    @property
    def tabs(self) -> list[Page]:
        if not self.browser_context:
            return []
        return list(self.browser_context.pages)

    async def start(self) -> "BrowserSession":
        self.browser_profile.prepare_user_data_dir()
        self.browser_profile.detect_display_configuration()
        if not self.playwright:
            self.playwright = await playwright_async_playwright().start()

        if self.browser_profile.user_data_dir and self.browser_profile.executable_path:
            logger.info(
                f"Attempting to launch persistent context with user_data_dir: {self.browser_profile.user_data_dir}"
            )
            persistent_kwargs = (
                self.browser_profile.kwargs_for_launch_persistent_context().model_dump(
                    exclude_none=True
                )
            )
            user_data_dir_to_launch = persistent_kwargs.pop("user_data_dir", None)
            if not user_data_dir_to_launch:
                raise ValueError(
                    "user_data_dir must be set in BrowserProfile for persistent context and was not found in launch kwargs."
                )
            user_data_dir_path = Path(user_data_dir_to_launch)
            self.browser_context = (
                await self.playwright.chromium.launch_persistent_context(
                    user_data_dir_path,
                    **persistent_kwargs,
                )
            )
            self.browser = self.browser_context.browser  # type: ignore
        else:
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
                self.browser_context = await self.browser.new_context(  # type: ignore
                    **self.browser_profile.kwargs_for_new_context().model_dump(
                        exclude_none=True
                    )
                )

        if not self.browser_context.pages:  # type: ignore
            self.agent_current_page = await self.browser_context.new_page()  # type: ignore
        else:
            self.agent_current_page = self.browser_context.pages[0]  # type: ignore

        self.initialized = True
        logger.info(
            f"BrowserSession started. Using {'persistent context at ' + str(self.browser_profile.user_data_dir) if self.browser_profile.user_data_dir and self.browser_profile.executable_path else 'new context'}."
        )
        return self

    async def stop(self):
        if not self.browser_profile.keep_alive:
            if self.browser_context:
                await self.browser_context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()  # type: ignore
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

    async def _wait_for_stable_network(self):
        pending_requests = set()
        last_activity = asyncio.get_event_loop().time()

        page = await self.get_current_page()

        # Define relevant resource types and content types
        RELEVANT_RESOURCE_TYPES = {
            "document",
            "stylesheet",
            "image",
            "font",
            "script",
            "iframe",
        }

        RELEVANT_CONTENT_TYPES = {
            "text/html",
            "text/css",
            "application/javascript",
            "image/",
            "font/",
            "application/json",
        }

        # Additional patterns to filter out
        IGNORED_URL_PATTERNS = {
            # Analytics and tracking
            "analytics",
            "tracking",
            "telemetry",
            "beacon",
            "metrics",
            # Ad-related
            "doubleclick",
            "adsystem",
            "adserver",
            "advertising",
            # Social media widgets
            "facebook.com/plugins",
            "platform.twitter",
            "linkedin.com/embed",
            # Live chat and support
            "livechat",
            "zendesk",
            "intercom",
            "crisp.chat",
            "hotjar",
            # Push notifications
            "push-notifications",
            "onesignal",
            "pushwoosh",
            # Background sync/heartbeat
            "heartbeat",
            "ping",
            "alive",
            # WebRTC and streaming
            "webrtc",
            "rtmp://",
            "wss://",
            # Common CDNs for dynamic content
            "cloudfront.net",
            "fastly.net",
        }

        async def on_request(request):
            # Filter by resource type
            if request.resource_type not in RELEVANT_RESOURCE_TYPES:
                return

            # Filter out streaming, websocket, and other real-time requests
            if request.resource_type in {
                "websocket",
                "media",
                "eventsource",
                "manifest",
                "other",
            }:
                return

            # Filter out by URL patterns
            url = request.url.lower()
            if any(pattern in url for pattern in IGNORED_URL_PATTERNS):
                return

            # Filter out data URLs and blob URLs
            if url.startswith(("data:", "blob:")):
                return

            # Filter out requests with certain headers
            headers = request.headers
            if headers.get("purpose") == "prefetch" or headers.get(
                "sec-fetch-dest"
            ) in [
                "video",
                "audio",
            ]:
                return

            nonlocal last_activity
            pending_requests.add(request)
            last_activity = asyncio.get_event_loop().time()
            # logger.debug(f'Request started: {request.url} ({request.resource_type})')

        async def on_response(response):
            request = response.request
            if request not in pending_requests:
                return

            # Filter by content type if available
            content_type = response.headers.get("content-type", "").lower()

            # Skip if content type indicates streaming or real-time data
            if any(
                t in content_type
                for t in [
                    "streaming",
                    "video",
                    "audio",
                    "webm",
                    "mp4",
                    "event-stream",
                    "websocket",
                    "protobuf",
                ]
            ):
                pending_requests.remove(request)
                return

            # Only process relevant content types
            if not any(ct in content_type for ct in RELEVANT_CONTENT_TYPES):
                pending_requests.remove(request)
                return

            # Skip if response is too large (likely not essential for page load)
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB
                pending_requests.remove(request)
                return

            nonlocal last_activity
            pending_requests.remove(request)
            last_activity = asyncio.get_event_loop().time()
            # logger.debug(f'Request resolved: {request.url} ({content_type})')

        # Attach event listeners
        page.on("request", on_request)
        page.on("response", on_response)

        now = asyncio.get_event_loop().time()
        try:
            # Wait for idle time
            start_time = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(0.1)
                now = asyncio.get_event_loop().time()
                if (
                    len(pending_requests) == 0
                    and (now - last_activity)
                    >= self.browser_profile.wait_for_network_idle_page_load_time
                ):
                    break
                if now - start_time > self.browser_profile.maximum_wait_page_load_time:
                    logger.debug(
                        f"Network timeout after {self.browser_profile.maximum_wait_page_load_time}s with {len(pending_requests)} "
                        f"pending requests: {[r.url for r in pending_requests]}"
                    )
                    break

        finally:
            # Clean up event listeners
            page.remove_listener("request", on_request)
            page.remove_listener("response", on_response)

        elapsed = now - start_time
        if elapsed > 1:
            logger.debug(
                f"💤 Page network traffic calmed down after {now - start_time:.2f} seconds"
            )

    async def _check_and_handle_navigation(self, page: Page) -> None:
        """Check if current page URL is allowed and handle if not."""
        if not self._is_url_allowed(page.url):
            logger.warning(f"⛔️  Navigation to non-allowed URL detected: {page.url}")
            try:
                await self.go_back()
            except Exception as e:
                logger.error(
                    f"⛔️  Failed to go back after detecting non-allowed URL: {str(e)}"
                )
            raise URLNotAllowedError(f"Navigation to non-allowed URL: {page.url}")

    async def _wait_for_page_and_frames_load(
        self, timeout_overwrite: float | None = None
    ):
        """
        Ensures page is fully loaded before continuing.
        Waits for either network to be idle or minimum WAIT_TIME, whichever is longer.
        Also checks if the loaded URL is allowed.
        """
        # Start timing
        start_time = time.time()

        # Wait for page load
        page = await self.get_current_page()
        try:
            await self._wait_for_stable_network()

            # Check if the loaded URL is allowed
            await self._check_and_handle_navigation(page)
        except URLNotAllowedError as e:
            raise e
        except Exception:
            logger.warning("⚠️  Page load failed, continuing...")

        # Calculate remaining time to meet minimum WAIT_TIME
        elapsed = time.time() - start_time
        remaining = max(
            (timeout_overwrite or self.browser_profile.minimum_wait_page_load_time)
            - elapsed,
            0,
        )

        # just for logging, calculate how much data was downloaded
        try:
            bytes_used = await page.evaluate(
                """
                () => {
                    let total = 0;
                    for (const entry of performance.getEntriesByType('resource')) {
                        total += entry.transferSize || 0;
                    }
                    for (const nav of performance.getEntriesByType('navigation')) {
                        total += nav.transferSize || 0;
                    }
                    return total;
                }
            """
            )
        except Exception:
            bytes_used = None

        tab_idx = self.tabs.index(page)
        if bytes_used is not None:
            logger.debug(
                f"➡️ Page navigation [{tab_idx}]{_log_pretty_url(page.url, 40)} used {bytes_used / 1024:.1f} KB in {elapsed:.2f}s, waiting +{remaining:.2f}s for all frames to finish"
            )
        else:
            logger.debug(
                f"➡️ Page navigation [{tab_idx}]{_log_pretty_url(page.url, 40)} took {elapsed:.2f}s, waiting +{remaining:.2f}s for all frames to finish"
            )

        # Sleep remaining time if needed
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def get_scroll_info(self, page: Page) -> tuple[int, int]:
        """Get scroll position information for the current page."""
        scroll_y = await page.evaluate("window.scrollY")
        viewport_height = await page.evaluate("window.innerHeight")
        total_height = await page.evaluate("document.documentElement.scrollHeight")
        pixels_above = scroll_y
        pixels_below = total_height - (scroll_y + viewport_height)
        return pixels_above, pixels_below

    @time_execution_async("--remove_highlights")
    async def remove_highlights(self):
        """
        Removes all highlight overlays and labels created by the highlightElement function.
        Handles cases where the page might be closed or inaccessible.
        """
        page = await self.get_current_page()
        try:
            await page.evaluate(
                """
                try {
                    // Remove the highlight container and all its contents
                    const container = document.getElementById('playwright-highlight-container');
                    if (container) {
                        container.remove();
                    }

                    // Remove highlight attributes from elements
                    const highlightedElements = document.querySelectorAll('[browser-user-highlight-id^="playwright-highlight-"]');
                    highlightedElements.forEach(el => {
                        el.removeAttribute('browser-user-highlight-id');
                    });
                } catch (e) {
                    console.error('Failed to remove highlights:', e);
                }
                """
            )
        except Exception as e:
            logger.debug(
                f"⚠  Failed to remove highlights (this is usually ok): {type(e).__name__}: {e}"
            )
            # Don't raise the error since this is not critical functionality

    async def get_state_summary(
        self, cache_clickable_elements_hashes: bool = True
    ) -> BrowserStateSummary:
        """Get a summary of the current browser state

        This method builds a BrowserStateSummary object that captures the current state
        of the browser, including url, title, tabs, screenshot, and DOM tree.

        Parameters:
        -----------
        cache_clickable_elements_hashes: bool
            If True, cache the clickable elements hashes for the current state.
            This is used to calculate which elements are new to the LLM since the last message,
            which helps reduce token usage.
        """
        await self._wait_for_page_and_frames_load()
        updated_state = await self._get_updated_state()

        # Find out which elements are new
        # Do this only if url has not changed
        if cache_clickable_elements_hashes:
            # if we are on the same url as the last state, we can use the cached hashes
            if (
                self._cached_clickable_element_hashes
                and self._cached_clickable_element_hashes.url == updated_state.url
            ):
                # Pointers, feel free to edit in place
                updated_state_clickable_elements = (
                    ClickableElementProcessor.get_clickable_elements(
                        updated_state.element_tree
                    )
                )

                for dom_element in updated_state_clickable_elements:
                    dom_element.is_new = (
                        ClickableElementProcessor.hash_dom_element(dom_element)
                        not in self._cached_clickable_element_hashes.hashes  # see which elements are new from the last state where we cached the hashes
                    )
            # in any case, we need to cache the new hashes
            self._cached_clickable_element_hashes = CachedClickableElementHashes(
                url=updated_state.url,
                hashes=ClickableElementProcessor.get_clickable_elements_hashes(
                    updated_state.element_tree
                ),
            )

        assert updated_state
        self._cached_browser_state_summary = updated_state

        # Save cookies if a file is specified
        if self.browser_profile.cookies_file:
            asyncio.create_task(self.save_cookies())

        return self._cached_browser_state_summary

    async def _get_updated_state(self, focus_element: int = -1) -> BrowserStateSummary:
        """Update and return state."""

        page = await self.get_current_page()

        # Check if current page is still valid, if not switch to another available page
        try:
            # Test if page is still accessible
            await page.evaluate("1")
        except Exception as e:
            logger.debug(
                f"👋  Current page is no longer accessible: {type(e).__name__}: {e}"
            )
            raise BrowserError("Browser closed: no valid pages available")

        try:
            await self.remove_highlights()
            dom_service = DomService(page)
            content = await dom_service.get_clickable_elements(
                focus_element=focus_element,
                viewport_expansion=self.browser_profile.viewport_expansion,
                highlight_elements=self.browser_profile.highlight_elements,
            )

            tabs_info = await self.get_tabs_info()

            # Get all cross-origin iframes within the page and open them in new tabs
            # mark the titles of the new tabs so the LLM knows to check them for additional content
            # unfortunately too buggy for now, too many sites use invisible cross-origin iframes for ads, tracking, youtube videos, social media, etc.
            # and it distracts the bot by opening a lot of new tabs
            # iframe_urls = await dom_service.get_cross_origin_iframes()
            # outer_page = self.agent_current_page
            # for url in iframe_urls:
            # 	if url in [tab.url for tab in tabs_info]:
            # 		continue  # skip if the iframe if we already have it open in a tab
            # 	new_page_id = tabs_info[-1].page_id + 1
            # 	logger.debug(f'Opening cross-origin iframe in new tab #{new_page_id}: {url}')
            # 	await self.create_new_tab(url)
            # 	tabs_info.append(
            # 		TabInfo(
            # 			page_id=new_page_id,
            # 			url=url,
            # 			title=f'iFrame opened as new tab, treat as if embedded inside page {outer_page.url}: {page.url}',
            # 			parent_page_url=outer_page.url,
            # 		)
            # 	)

            screenshot_b64 = await self.take_screenshot()
            pixels_above, pixels_below = await self.get_scroll_info(page)

            self.browser_state_summary = BrowserStateSummary(
                element_tree=content.element_tree,
                selector_map=content.selector_map,
                url=page.url,
                title=await page.title(),
                tabs=tabs_info,
                screenshot=screenshot_b64,
                pixels_above=pixels_above,
                pixels_below=pixels_below,
            )

            return self.browser_state_summary
        except Exception as e:
            logger.error(f"❌  Failed to update state: {e}")
            # Return last known good state if available
            if hasattr(self, "browser_state_summary"):
                return self.browser_state_summary
            raise

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
            return False
        return any(
            (
                domain.endswith(allowed_domain.lstrip("*."))
                if allowed_domain.startswith("*.")
                else domain == allowed_domain
            )
            for allowed_domain in self.browser_profile.allowed_domains  # type: ignore
        )

    async def get_tabs_info(self) -> List[TabInfo]:
        if not self.browser_context:
            return []
        tabs_info = []
        for i, page in enumerate(self.browser_context.pages):
            try:
                tabs_info.append(
                    TabInfo(page_id=i, url=page.url, title=await page.title())
                )
            except Exception:
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
                self.agent_current_page = None  # type: ignore
            await self.get_current_page()


# --- Action Models (Simplified) ---
class ActionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_index(self) -> Optional[int]:
        for _, value in self:
            if isinstance(value, dict) and "index" in value:
                return value["index"]
            if hasattr(value, "index") and value.index is not None:  # type: ignore
                return value.index  # type: ignore
        return None

    def set_index(self, index: int):
        for field_name, _ in self:
            value = getattr(self, field_name)
            if isinstance(value, dict) and "index" in value:
                value["index"] = index
                setattr(self, field_name, value)
                return
            if hasattr(value, "index"):  # type: ignore
                value.index = index  # type: ignore
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
        return f"{self.description}: {self.name}({self.param_model.__name__})"


class ActionRegistry(BaseModel):
    actions: Dict[str, RegisteredAction] = {}

    def get_prompt_description(self, page: Optional[Page] = None) -> str:
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
            if (
                page
                and action_info.domains
                and not any(
                    urlparse(page.url).hostname.endswith(d.lstrip("*."))  # type: ignore
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
        DynamicActionModel = create_model(
            "DynamicActionModel", __base__=ActionModel, **fields
        )  # type: ignore
        return DynamicActionModel


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
                data: output_model  # type: ignore
                success: bool

            self.registry.actions["done"] = RegisteredAction(
                name="done",
                description="Completes the task with custom output",
                function=self._custom_done_action_func,  # type: ignore
                param_model=CustomDoneAction,
            )

    async def _custom_done_action_func(self, params: BaseModel):
        return ActionResult(
            is_done=True,
            success=params.success,  # type: ignore
            extracted_content=params.data.model_dump_json(),  # type: ignore
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
                    if name not in ["browser_session", "page_extraction_llm", "context"]
                }
                actual_param_model = create_model(f"{func.__name__}Params", **fields)  # type: ignore
            self.registry.actions[func.__name__] = RegisteredAction(
                name=func.__name__,
                description=description,
                function=func,
                param_model=actual_param_model,  # type: ignore
                **kwargs,  # type: ignore
            )
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
                    )  # type: ignore
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
        async def done(params: DoneAction):
            return ActionResult(
                is_done=True, success=params.success, extracted_content=params.text
            )

        @self.action("Open URL in a new tab", param_model=OpenTabAction)
        async def open_tab(params: OpenTabAction, browser_session: BrowserSession):
            if not browser_session.browser_context:
                await browser_session.start()
            new_page = await browser_session.browser_context.new_page()  # type: ignore
            await new_page.goto(params.url, wait_until="domcontentloaded")
            browser_session.agent_current_page = new_page
            return ActionResult(
                extracted_content=f"Opened new tab with URL: {params.url}",
                include_in_memory=True,
            )

        @self.action("Close an existing tab by its ID", param_model=CloseTabAction)
        async def close_tab_action(
            params: CloseTabAction, browser_session: BrowserSession
        ):  # Renamed to avoid conflict
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
            "Extract content from the current page based on a goal",  # Description from your file
            param_model=ExtractPageContentAction,
        )
        async def extract_content(
            params: ExtractPageContentAction,
            browser_session: BrowserSession,
            page_extraction_llm: BaseChatModel,  # This is an LLM call within the action
        ):
            page = await browser_session.get_current_page()
            extracted_data_summary = (
                f"Attempted to extract: '{params.goal}'."  # Default summary
            )

            try:
                # Try to get cleaner text from common main content areas first
                main_content_selectors = [
                    "main",
                    "article",
                    "[role='main']",
                    "body",
                ]  # Prioritized selectors
                text_content = ""
                logger.info(f"Attempting to extract text for goal: '{params.goal}'")

                for selector in main_content_selectors:
                    try:
                        content_elements = page.locator(selector)
                        count = await content_elements.count()
                        if count > 0:
                            # Prioritize visible elements if multiple are found
                            visible_element_text = ""
                            for i in range(count):
                                element = content_elements.nth(i)
                                if await element.is_visible(
                                    timeout=1000
                                ):  # Short timeout for visibility check
                                    visible_element_text = await element.inner_text(
                                        timeout=5000
                                    )  # 5 sec timeout
                                    if visible_element_text.strip():
                                        break  # Use the first visible element's text

                            if visible_element_text.strip():
                                text_content = visible_element_text.strip()
                                logger.info(
                                    f"Successfully extracted text using selector '{selector}'. Length: {len(text_content)}"
                                )
                                break  # Got content from a prioritized selector
                            else:
                                logger.debug(
                                    f"Selector '{selector}' found elements, but no visible text obtained."
                                )
                        else:
                            logger.debug(f"Selector '{selector}' not found on page.")
                    except PlaywrightTimeoutError:
                        logger.warning(
                            f"Timeout extracting inner_text from selector '{selector}'."
                        )
                        continue
                    except Exception as el_err:
                        logger.debug(
                            f"Error processing selector '{selector}': {el_err}"
                        )
                        continue

                if not text_content.strip():
                    logger.warning(
                        "No text content extracted from main content selectors. Full page text might be too noisy or extraction might fail."
                    )
                    # As a last resort, if no specific content area worked, you could try a limited body text or indicate failure.
                    # For this iteration, we'll proceed with potentially empty text_content if nothing specific was found,
                    # letting the sub-LLM handle the "nothing found" case.

            except Exception as e:
                logger.error(
                    f"Critical error during page text scraping for 'extract_content': {type(e).__name__} - {e}",
                    exc_info=True,
                )
                return ActionResult(
                    error=f"Failed to scrape page content: {type(e).__name__}",
                    include_in_memory=True,
                )

            # Refine the goal for the page_extraction_llm
            # The original params.goal might be something like "Extract the titles and links of 5 news articles..."
            # The sub-LLM should just focus on finding titles and links based on the main goal.
            simplified_extraction_goal = params.goal
            if "extract the titles and links of 5 news articles" in params.goal.lower():
                simplified_extraction_goal = "titles and links of news articles about COVID-19 in Vietnam"  # Be more direct

            # Limit the text sent to the sub-LLM to a reasonable length
            # Increased slightly, but be mindful of token limits and sub-LLM performance
            max_text_for_sub_llm = 10000
            if len(text_content) > max_text_for_sub_llm:
                logger.info(
                    f"Truncating text_content for page_extraction_llm from {len(text_content)} to {max_text_for_sub_llm} chars."
                )
                text_to_process = text_content[:max_text_for_sub_llm]
            else:
                text_to_process = text_content

            if not text_to_process.strip():
                logger.warning(
                    f"No text content available to send to page_extraction_llm for goal: '{simplified_extraction_goal}'"
                )
                extracted_data_summary = f"No text content found on the page to process for: '{simplified_extraction_goal}'."
            elif page_extraction_llm:
                # Construct a more specific prompt for the sub-LLM
                sub_llm_prompt = (
                    f"Review the following TEXT_TO_PROCESS and extract information for the goal: '{simplified_extraction_goal}'.\n"
                    "If the goal involves finding multiple items (e.g., articles, products), list each item clearly. "
                    "For news articles, please extract the title and the direct URL (link/href) for each article found. "
                    "Present the findings as a clear, well-structured list. "
                    "If no relevant information or articles are found, explicitly state 'No specific information or articles found matching the goal.'\n\n"
                    "TEXT_TO_PROCESS:\n"
                    f'"""{text_to_process}"""'
                )
                try:
                    logger.info(
                        f"Sending to page_extraction_llm. Effective Goal: '{simplified_extraction_goal}'. Text length: {len(text_to_process)}"
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
                        extracted_data_summary = f"Sub-LLM found no specific articles/information for: '{simplified_extraction_goal}'."
                    else:
                        # Success, use the direct data from sub-LLM
                        extracted_data_summary = extracted_data
                    logger.info(
                        f"page_extraction_llm raw response snippet: {extracted_data_summary[:300]}..."
                    )

                except Exception as e:
                    logger.error(
                        f"page_extraction_llm call failed: {type(e).__name__} - {e}",
                        exc_info=True,
                    )
                    extracted_data_summary = f"Error during sub-LLM extraction for '{simplified_extraction_goal}': {type(e).__name__}."
            else:
                extracted_data_summary = (
                    "page_extraction_llm not configured; cannot process extracted text."
                )

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
        action_name = ""
        dumped_action = action.model_dump(exclude_unset=True)
        if not dumped_action:
            return ActionResult(error="No action specified in the model.")
        action_name = list(dumped_action.keys())[0]
        params_obj = dumped_action[action_name]
        if action_name not in self.registry.actions:
            return ActionResult(error=f"Action '{action_name}' not found.")
        registered_action = self.registry.actions[action_name]
        action_kwargs: Dict[str, Any] = {}
        sig = inspect.signature(registered_action.function)
        if "browser_session" in sig.parameters:
            action_kwargs["browser_session"] = browser_session
        if "page_extraction_llm" in sig.parameters:
            action_kwargs["page_extraction_llm"] = page_extraction_llm
        try:
            if isinstance(params_obj, BaseModel):
                validated_params = params_obj
            elif isinstance(params_obj, dict):
                validated_params = registered_action.param_model(**params_obj)
            else:
                validated_params = registered_action.param_model()
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
                else:
                    result = await registered_action.function(
                        **validated_params.model_dump(exclude_none=True),
                        **action_kwargs,
                    )  # type: ignore
            else:
                result = await registered_action.function(
                    **validated_params.model_dump(exclude_none=True), **action_kwargs
                )  # type: ignore
            if isinstance(result, ActionResult):
                return result
            if isinstance(result, str):
                return ActionResult(extracted_content=result)
            return ActionResult()
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
        """Check if this is the last step"""
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
        if isinstance(message.content, str):
            content_str = message.content
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_str += item["text"]
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    content_str += " [IMAGE] "
        if hasattr(message, "tool_calls") and message.tool_calls:  # type: ignore
            content_str += str(message.tool_calls)  # type: ignore
        char_tokens = len(content_str) // self.settings.estimated_characters_per_token
        image_count = content_str.count("[IMAGE]")
        return char_tokens + (image_count * self.settings.image_tokens)

    def _add_message_with_tokens(
        self,
        message: BaseMessage,
        position: Optional[int] = None,
        message_type: Optional[str] = None,
    ):
        if self.settings.sensitive_data and isinstance(message.content, str):
            temp_content = message.content
            for placeholder, real_value in self.settings.sensitive_data.items():
                if real_value:
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
            message.content = new_content_list  # type: ignore
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

        example_current_state = {
            "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Shortly state why/why not",
            "memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz",
            "next_goal": "What needs to be done with the next immediate action",
        }
        example_action_list = [{"one_action_name": {"parameter_name": "value"}}]
        example_args_for_llm_tool_call = {
            "current_state": example_current_state,
            "action": example_action_list,
        }

        self._add_message_with_tokens(
            HumanMessage(
                content=f"Example of the JSON structure you should provide for the 'AgentOutput' tool's arguments:\n```json\n{json.dumps(example_args_for_llm_tool_call, indent=2)}\n```"
            ),
            message_type="init",
        )

        # Corrected example AIMessage for Langchain's ToolCall structure
        example_tool_call_id = "tool_call_example_id"
        example_aimessage_with_tool_call = AIMessage(
            content="I will now perform the action.",
            tool_calls=[
                {
                    "id": example_tool_call_id,
                    "name": "AgentOutput",
                    "args": example_args_for_llm_tool_call,  # Arguments as a dict
                }
            ],
        )
        self._add_message_with_tokens(
            example_aimessage_with_tool_call, message_type="init"
        )
        self.add_tool_message(
            content="Example tool call processed.",
            tool_call_id=example_tool_call_id,
            message_type="init",
        )

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
        self.task = new_task
        self._add_message_with_tokens(
            HumanMessage(
                content=f'Your new ultimate task is: """{new_task}""". Consider previous context.'
            )
        )

    def add_state_message(
        self,
        browser_state_summary: BrowserStateSummary,
        result: list[ActionResult] | None = None,
        step_info: AgentStepInfo | None = None,
        use_vision=True,
    ) -> None:
        """Add browser state as human message"""

        # if keep in memory, add to directly to history and add state without result
        if result:
            for r in result:
                if r.include_in_memory:
                    if r.extracted_content:
                        msg = HumanMessage(
                            content="Action result: " + str(r.extracted_content)
                        )
                        self._add_message_with_tokens(msg)
                    if r.error:
                        # if endswith \n, remove it
                        if r.error.endswith("\n"):
                            r.error = r.error[:-1]
                        # get only last line of error
                        last_line = r.error.split("\n")[-1]
                        msg = HumanMessage(content="Action error: " + last_line)
                        self._add_message_with_tokens(msg)
                    result = None  # if result in history, we dont want to add it again

        # otherwise add state message and result to next message (which will not stay in memory)
        assert browser_state_summary
        state_message = AgentMessagePrompt(
            browser_state_summary=browser_state_summary,
            result=result,
            include_attributes=self.settings.include_attributes,
            step_info=step_info,
        ).get_user_message(use_vision)
        self._add_message_with_tokens(state_message)

    def add_model_output(self, model_output: "AgentOutput"):
        tool_call_id = str(self.state.tool_id)
        # Corrected structure for Langchain's ToolCall
        tool_calls = [
            {
                "id": tool_call_id,
                "name": "AgentOutput",
                "args": model_output.model_dump(
                    exclude_unset=True, exclude_none=True
                ),  # Arguments as dict
            }
        ]
        ai_message = AIMessage(content="", tool_calls=tool_calls)
        self._add_message_with_tokens(ai_message)
        self.add_tool_message(content="Action processed.", tool_call_id=tool_call_id)

    def add_tool_message(
        self, content: str, tool_call_id: str, message_type: Optional[str] = None
    ):
        tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
        self._add_message_with_tokens(tool_message, message_type=message_type)
        self.state.tool_id += 1

    def get_messages(self) -> List[BaseMessage]:
        return [m.message for m in self.state.history.messages]

    def _ensure_token_limit(self):
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
            else:
                break

    def cut_messages(self):
        pass


# --- Agent Prompts (Modified) ---
BROWSER_USE_SYSTEM_PROMPT_TEMPLATE = """
You are an AI agent designed to automate browser tasks. Your goal is to accomplish the ultimate task following the rules.

# Input Format

Task
Previous steps
Current URL
Open Tabs
Interactive Elements
[index]<type>text</type>

- index: Numeric identifier for interaction
- type: HTML element type (button, input, etc.)
- text: Element description
  Example:
  [33]<div>User form</div>
  \t*[35]*<button aria-label='Submit form'>Submit</button>

- Only elements with numeric indexes in [] are interactive
- (stacked) indentation (with \t) is important and means that the element is a (html) child of the element above (with a lower index)
- Elements with \* are new elements that were added after the previous step (if url has not changed)

# Response Rules

1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {{"current_state": {{"evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Shortly state why/why not",
   "memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz",
   "next_goal": "What needs to be done with the next immediate action"}},
   "action":[{{"one_action_name": {{// action-specific parameter}}}}, // ... more actions in sequence]}}

2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item. Use maximum {max_actions_per_step} actions per sequence.
Common action sequences:

- Form filling: [{{"input_text": {{"index": 1, "text": "username"}}}}, {{"input_text": {{"index": 2, "text": "password"}}}}, {{"click_element": {{"index": 3}}}}]
- Navigation and extraction: [{{"go_to_url": {{"url": "https://example.com"}}}}, {{"extract_content": {{"goal": "extract the names"}}}}]
- Actions are executed in the given order
- If the page changes after an action, the sequence is interrupted and you get the new state.
- Only provide the action sequence until an action which changes the page state significantly.
- Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page
- only use multiple actions if it makes sense.

3. ELEMENT INTERACTION:

- Only use indexes of the interactive elements

4. NAVIGATION & ERROR HANDLING:

- If no suitable elements exist, use other functions to complete the task
- If stuck, try alternative approaches - like going back to a previous page, new search, new tab etc.
- Handle popups/cookies by accepting or closing them
- Use scroll to find elements you are looking for
- If you want to research something, open a new tab instead of using the current tab
- If captcha pops up, try to solve it - else try a different approach
- If the page is not fully loaded, use wait action

5. TASK COMPLETION:

- Use the done action as the last action as soon as the ultimate task is complete
- Dont use "done" before you are done with everything the user asked you, except you reach the last step of max_steps.
- If you reach your last step, use the done action even if the task is not fully finished. Provide all the information you have gathered so far. If the ultimate task is completely finished set success to true. If not everything the user asked for is completed set success in done to false!
- If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
- Don't hallucinate actions
- Make sure you include everything you found out for the ultimate task in the done text parameter. Do not just say you are done, but include the requested information of the task.

6. VISUAL CONTEXT:

- When an image is provided, use it to understand the page layout
- Bounding boxes with labels on their top right corner correspond to element indexes

7. Form filling:

- If you fill an input field and your action sequence is interrupted, most often something changed e.g. suggestions popped up under the field.

8. Long tasks:

- Keep track of the status and subresults in the memory.
- You are provided with procedural memory summaries that condense previous task history (every N steps). Use these summaries to maintain context about completed actions, current progress, and next steps. The summaries appear in chronological order and contain key information about navigation history, findings, errors encountered, and current state. Refer to these summaries to avoid repeating actions and to ensure consistent progress toward the task goal.

9. Extraction:

- If your task is to find information - call extract_content on the specific pages to get and store the information.
  Your responses must be always JSON with the specified format.

Available actions:
{action_description}
"""


class SystemPrompt:
    def __init__(
        self,
        action_description: str,
        max_actions_per_step: int = 20,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
    ):
        prompt = ""
        if override_system_message:
            prompt = override_system_message
        else:
            prompt = BROWSER_USE_SYSTEM_PROMPT_TEMPLATE.format(
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
        browser_state_summary: "BrowserStateSummary",
        result: list["ActionResult"] | None = None,
        include_attributes: list[str] | None = None,
        step_info: Optional[Any] = None,
    ):
        self.state: "BrowserStateSummary" = browser_state_summary
        self.result = result
        self.include_attributes = include_attributes or []
        self.step_info = step_info
        assert self.state

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        elements_text = self.state.element_tree.clickable_elements_to_string(
            include_attributes=self.include_attributes
        )

        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0

        if elements_text != "":
            if has_content_above:
                elements_text = f"... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}"
            else:
                elements_text = f"[Start of page]\n{elements_text}"
            if has_content_below:
                elements_text = f"{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ..."
            else:
                elements_text = f"{elements_text}\n[End of page]"
        else:
            elements_text = "empty page"

        if self.step_info:
            step_info_description = f"Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}"
        else:
            step_info_description = ""
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        step_info_description += f"Current date and time: {time_str}"

        state_description = f"""
[Task history memory ends]
[Current state starts here]
The following is one-time information - if you need to remember it write it to memory:
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements from top layer of the current page inside the viewport:
{elements_text}
{step_info_description}
"""

        if self.result:
            for i, result in enumerate(self.result):
                if result.extracted_content:
                    state_description += f"\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}"
                if result.error:
                    # only use last line of error
                    error = result.error.split("\n")[-1]
                    state_description += (
                        f"\nAction error {i + 1}/{len(self.result)}: ...{error}"
                    )

        if self.state.screenshot and use_vision is True:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_description},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.state.screenshot}"
                        },  # , 'detail': 'low'
                    },
                ]
            )

        return HumanMessage(content=state_description)


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
        DynamicAgentOutput = create_model(
            "DynamicAgentOutput",
            __base__=AgentOutput,
            action=(List[custom_actions_model], Field(..., min_length=1)),
            __module__=AgentOutput.__module__,
        )  # type: ignore
        return DynamicAgentOutput


class AgentSettings(BaseModel):
    use_vision: bool = True
    save_conversation_path: Optional[str] = None
    max_failures: int = 3
    retry_delay: int = 10
    override_system_message: Optional[str] = None
    extend_system_message: Optional[str] = None
    max_input_tokens: int = 128000  # Your value
    validate_output: bool = False
    message_context: Optional[str] = None
    generate_gif: Union[bool, str] = False
    available_file_paths: Optional[List[str]] = None
    max_actions_per_step: int = 5  # Your value
    tool_calling_method: Optional[
        Literal["function_calling", "json_mode", "raw", "auto", "tools"]
    ] = "auto"
    page_extraction_llm: Optional[BaseChatModel] = None
    planner_llm: Optional[BaseChatModel] = None  # Keep if you have it
    planner_interval: int = 1  # Keep if you have it
    is_planner_reasoning: bool = False  # Keep if you have it
    save_playwright_script_path: Optional[str] = None  # Keep if you have it
    extend_planner_system_message: Optional[str] = None  # Keep if you have it
    # Add this new setting:
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


class AgentHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_output: Optional[AgentOutput] = None
    result: List[ActionResult]
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AgentHistoryList(BaseModel):
    history: List[AgentHistory] = Field(default_factory=list)

    def is_done(self) -> bool:
        return bool(
            self.history
            and self.history[-1].result
            and self.history[-1].result[-1].is_done
        )

    def is_successful(self) -> Optional[bool]:
        return self.history[-1].result[-1].success if self.is_done() else None

    def final_result(self) -> Optional[str]:
        return self.history[-1].result[-1].extracted_content if self.is_done() else None


AgentState.model_rebuild()


# --- Agent Service (Modified) ---
class Agent(Generic[Context]):
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: Optional[BrowserSession] = None,
        controller: Optional[Controller[Context]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        use_vision: bool = True,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
        max_input_tokens: int = 120000,
        max_actions_per_step: int = 5,
        page_extraction_llm: Optional[BaseChatModel] = None,
        injected_agent_state: Optional[AgentState] = None,
        context: Optional[Context] = None,
        agent_settings: Optional[AgentSettings] = None,
        browser_profile: Optional[BrowserProfile] = None,
    ):
        self.task = task
        self.llm = llm
        self.controller = controller or Controller()
        self.sensitive_data = sensitive_data
        self.version = "main.py-replica-0.4"  # Updated version
        self.settings = (
            agent_settings
            or AgentSettings(
                use_vision=use_vision,  # from __init__ args
                override_system_message=override_system_message,  # from __init__ args
                extend_system_message=extend_system_message,  # from __init__ args
                max_input_tokens=max_input_tokens,  # from __init__ args
                max_actions_per_step=max_actions_per_step,  # from __init__ args
                page_extraction_llm=page_extraction_llm or llm,  # from __init__ args
                interrupt_on_page_change_in_multi_act=True,  # Default or ensure it's in provided agent_settings
            )
        )
        self.state = injected_agent_state or AgentState()
        if browser_session:
            self.browser_session = browser_session
        else:
            self.browser_session = BrowserSession(
                browser_profile=browser_profile or DEFAULT_BROWSER_PROFILE
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
                f"LLM {self.llm.__class__.__name__} may not support bind_tools with method '{self.tool_calling_method}'. Tool calling might fail."
            )
        active_browser_profile = self.browser_session.browser_profile
        message_manager_settings = MessageManagerSettings(
            max_input_tokens=self.settings.max_input_tokens,
            include_attributes=active_browser_profile.include_attributes,
            message_context=self.settings.message_context,
            sensitive_data=self.sensitive_data,  # type: ignore
            available_file_paths=self.settings.available_file_paths,
        )
        self._message_manager = MessageManager(
            task=task,
            system_message=SystemPrompt(
                action_description=self.controller.registry.get_prompt_description(),
                max_actions_per_step=self.settings.max_actions_per_step,
                override_system_message=self.settings.override_system_message,
                extend_system_message=self.settings.extend_system_message,
            ).get_system_message(),
            settings=message_manager_settings,
            state=self.state.message_manager_state,
        )
        self._external_pause_event = asyncio.Event()
        self._external_pause_event.set()

    def _convert_initial_actions(
        self, actions: List[Dict[str, Dict[str, Any]]]
    ) -> List[ActionModel]:
        converted_actions = []
        for action_dict in actions:
            try:
                converted_actions.append(self.ActionModelType(**action_dict))
            except ValidationError as e:
                logger.error(f"Failed to validate initial action {action_dict}: {e}")
        return converted_actions

    async def get_next_action(self, input_messages: List[BaseMessage]) -> AgentOutput:
        try:
            if self.tool_calling_method in ["tools", "function_calling"]:
                llm_with_tool = self.llm.bind_tools(
                    tools=[self.AgentOutputType],
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.AgentOutputType.__name__},
                    },
                )
            elif self.tool_calling_method == "json_mode":
                llm_with_tool = self.llm
            else:
                llm_with_tool = self.llm
        except Exception as e:
            logger.error(
                f"Failed to configure LLM for tool calling/JSON output with method '{self.tool_calling_method}'. Error: {e}"
            )
            llm_with_tool = self.llm
        raw_response = await llm_with_tool.ainvoke(input_messages)
        if (
            self.tool_calling_method in ["tools", "function_calling"]
            and hasattr(raw_response, "tool_calls")
            and raw_response.tool_calls
        ):  # type: ignore
            tool_call = raw_response.tool_calls[0]  # type: ignore
            tool_name = None
            tool_arguments_str = None
            if tool_call.get("type") == "function" and "function" in tool_call:
                tool_function_data = tool_call["function"]
                tool_name = tool_function_data.get("name")
                tool_arguments_str = tool_function_data.get("arguments")
            else:
                tool_name = tool_call.get("name")
                tool_arguments_str = tool_call.get("args")
            if (
                not tool_name
                or tool_name.lower() != self.AgentOutputType.__name__.lower()
            ):  # type: ignore
                logger.error(f"LLM called unexpected tool: {tool_name}")
                raise ValueError(f"LLM called unexpected tool: {tool_name}")
            try:
                if isinstance(tool_arguments_str, str):
                    try:
                        args_dict = json.loads(tool_arguments_str)
                        model_output = self.AgentOutputType(**args_dict)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse JSON string from tool call arguments: {tool_arguments_str}. Error: {e}"
                        )
                        raise ValueError(
                            f"Could not parse tool arguments: {tool_arguments_str}"
                        ) from e
                elif isinstance(tool_arguments_str, dict):
                    model_output = self.AgentOutputType(**tool_arguments_str)
                else:
                    raise ValueError(
                        f"Tool arguments are not a valid string or dict: {tool_arguments_str}"
                    )
            except ValidationError as e:
                logger.error(
                    f"Failed to validate AgentOutput from tool call arguments: {e}. Args: {tool_arguments_str}"
                )
                raise ValueError(
                    f"Could not parse tool arguments for AgentOutput: {tool_arguments_str}"
                ) from e
        else:
            if isinstance(raw_response.content, str):
                try:
                    content_str = raw_response.content
                    if content_str.startswith("```json"):
                        content_str = content_str[7:]
                    if content_str.endswith("```"):
                        content_str = content_str[:-3]
                    content_str = content_str.strip()
                    parsed_json = json.loads(content_str)
                    model_output = self.AgentOutputType(**parsed_json)
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
        log_response(model_output)
        return model_output

    # In your Agent class in main.py

    async def multi_act(self, actions: List[ActionModel]) -> List[ActionResult]:
        results = []
        initial_hashes_url = None
        initial_hashes_set = set()

        # These are the hashes and URL from *before* the LLM made the current plan (actions list).
        # This information should be current in the browser_session's cache
        # due to the get_state_summary call that preceded the LLM invocation.
        if self.browser_session._cached_clickable_element_hashes:
            initial_hashes_url = (
                self.browser_session._cached_clickable_element_hashes.url
            )
            initial_hashes_set = (
                self.browser_session._cached_clickable_element_hashes.hashes
            )

        for i, action_model_instance in enumerate(actions):
            if self.state.stopped:  # Check if agent was stopped externally
                logger.info("Agent stop requested during multi_act.")
                results.append(
                    ActionResult(
                        error="Agent stopped during action sequence.",
                        include_in_memory=True,
                    )
                )
                break

            # --- Parameter validation for action_model_instance (copied from your existing multi_act if you had it) ---
            if not isinstance(action_model_instance, self.ActionModelType):
                logger.error(
                    f"Action {i} is not an instance of the expected ActionModel. Got: {type(action_model_instance)}"
                )
                if isinstance(action_model_instance, dict):
                    try:
                        action_model_instance = self.ActionModelType(
                            **action_model_instance  # type: ignore
                        )
                    except ValidationError as e:
                        results.append(
                            ActionResult(
                                error=f"Invalid action structure for action {i}: {e}"
                            )
                        )
                        break  # Stop processing further actions in this sequence
                else:
                    results.append(
                        ActionResult(
                            error=f"Action {i} is not a valid ActionModel or dictionary."
                        )
                    )
                    break  # Stop processing further actions in this sequence
            # --- End of parameter validation ---

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
                break  # Stop on task completion or if an action itself errors

            # If not the last action in the *planned* sequence, check for page changes
            if i < len(actions) - 1:
                # Wait a bit for page to potentially settle after the action
                # This wait might already be part of _wait_for_page_and_frames_load
                # called by get_state_summary, but an explicit short wait here can be defensive.
                await asyncio.sleep(
                    self.browser_session.browser_profile.wait_between_actions / 2
                )  # Shorter than full wait_between_actions

                if self.settings.interrupt_on_page_change_in_multi_act:
                    # Get fresh state summary. This call will also update
                    # self.browser_session._cached_clickable_element_hashes for the next main agent loop.
                    # Crucially, it gives us the state *after* the action `i` was performed.
                    fresh_state_summary = await self.browser_session.get_state_summary(
                        cache_clickable_elements_hashes=True  # Ensure the main cache is updated
                    )
                    current_url_after_action = fresh_state_summary.url

                    # These are the hashes of the page *after* the action was performed
                    new_hashes_after_action = (
                        self.browser_session._cached_clickable_element_hashes.hashes
                        if self.browser_session._cached_clickable_element_hashes
                        else set()
                    )

                    url_changed_from_original_plan_state = (
                        current_url_after_action != initial_hashes_url
                    )

                    # Compare DOM hashes only if the URL is still the one the LLM's plan was based on.
                    # If URL changed, that's a definite break.
                    # If URL is same as initial, check if DOM changed.
                    dom_changed_on_original_plan_url = (
                        not url_changed_from_original_plan_state
                        and new_hashes_after_action != initial_hashes_set
                    )

                    if url_changed_from_original_plan_state:
                        logger.info(
                            f"URL changed during multi_act (from '{initial_hashes_url}' to '{current_url_after_action}'), interrupting action sequence."
                        )
                        break  # Interrupt and re-prompt LLM
                    if dom_changed_on_original_plan_url:
                        logger.info(
                            f"DOM structure changed on URL '{initial_hashes_url}' during multi_act (hashes differ), interrupting action sequence."
                        )
                        break  # Interrupt and re-prompt LLM

            # If it was the last action in the sequence, or if only one action was planned,
            # apply the standard wait_between_actions (or a portion of it) before finishing multi_act.
            # This ensures there's a pause before the agent loop might immediately re-evaluate state.
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

    async def step(self, step_info: Optional[Any] = None):
        logger.info(f"--- Step {self.state.n_steps} ---")
        browser_state_summary = None
        model_output = None
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
            self._message_manager.add_model_output(model_output)  # type: ignore
            result = await self.multi_act(model_output.action)  # type: ignore
            self.state.last_result = result
            self.state.consecutive_failures = 0
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result
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
                )  # type: ignore
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
            class AgentStepInfo:
                step_number: int
                max_steps: int

            await self.step(
                step_info=AgentStepInfo(step_number=step_num, max_steps=max_steps)
            )
            if on_step_end:
                await on_step_end(self)  # type: ignore
            if self.state.history.is_done():
                logger.info("✅ Task marked as done.")
                break
        else:
            logger.info(f"Max steps ({max_steps}) reached.")
            if not self.state.history.is_done():
                done_params = DoneAction(
                    text="Max steps reached, task may be incomplete.", success=False
                )
                final_action_instance = self.ActionModelType(done=done_params)
                final_agent_brain = AgentBrain(
                    evaluation_previous_goal="Max steps reached",
                    memory="N/A",
                    next_goal="N/A",
                )
                final_model_output = self.AgentOutputType(
                    current_state=final_agent_brain, action=[final_action_instance]
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
                    json.dump(self.state.history.model_dump(), f, indent=2)
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


def log_response(response: AgentOutput) -> None:
    emoji = "🤷"
    if response.current_state and response.current_state.evaluation_previous_goal:
        if "Success" in response.current_state.evaluation_previous_goal:
            emoji = "👍"
        elif "Failed" in response.current_state.evaluation_previous_goal:
            emoji = "⚠"
        logger.info(f"{emoji} Eval: {response.current_state.evaluation_previous_goal}")
        logger.info(f"🧠 Memory: {response.current_state.memory}")
        logger.info(f"🎯 Next goal: {response.current_state.next_goal}")
    if response.action:
        for i, action_model_instance in enumerate(response.action):
            action_dict = action_model_instance.model_dump(exclude_unset=True)
            action_name = (
                list(action_dict.keys())[0] if action_dict else "unknown_action"
            )
            action_params = action_dict.get(action_name, {})
            logger.info(
                f"🛠️ Action {i + 1}/{len(response.action)}: {action_name}({json.dumps(action_params)})"
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

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    task = "Go to google and list 5 news about covid 19 in Vietnam"
    persistent_profile_path = os.getenv(
        "PERSISTENT_PROFILE_PATH",
    )
    chrome_exe_path = os.getenv(
        "CHROME_EXE_PATH",
    )
    other_browser_args = ["--start-maximized"]
    browser_profile = BrowserProfile(
        user_data_dir=persistent_profile_path,
        executable_path=chrome_exe_path,
        args=other_browser_args,
        headless=False,
        include_attributes=[
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
    )
    browser_session = BrowserSession(browser_profile=browser_profile)
    agent_settings = AgentSettings(
        use_vision=True, max_actions_per_step=5, tool_calling_method="tools"
    )
    agent = Agent(
        task=task,
        llm=llm,
        browser_session=browser_session,
        agent_settings=agent_settings,
    )
    try:
        print(f"Starting agent with task: {task}")
        history = await agent.run(max_steps=20)
        print("\n--- Agent Run History ---")
        if history.history:
            for i, item in enumerate(history.history):
                print(f"\n--- History Step {i + 1} ---")
                if item.model_output and isinstance(item.model_output, AgentOutput):
                    current_state = item.model_output.current_state
                    if current_state:
                        print(
                            f"  LLM Thought: Eval='{current_state.evaluation_previous_goal}' -> Memory='{current_state.memory}' -> NextGoal='{current_state.next_goal}'"
                        )
                    for action_item_model in item.model_output.action:
                        action_dump = action_item_model.model_dump(exclude_unset=True)
                        action_name = (
                            list(action_dump.keys())[0] if action_dump else "unknown"
                        )
                        action_params = action_dump.get(action_name, {})
                        print(
                            f"  LLM Action: {action_name}({json.dumps(action_params)})"
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


if __name__ == "__main__":
    __main__()
