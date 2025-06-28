import asyncio
import base64
import json  # Added import for saving cookies
import logging
import os
import time  # Added import
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import (
    Browser as PlaywrightBrowser,
)
from playwright.async_api import (
    BrowserContext as PlaywrightBrowserContext,
)
from playwright.async_api import (
    Page,
)
from playwright.async_api import (
    async_playwright as playwright_async_playwright,
)
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)
from typing_extensions import (  # Changed from typing to typing_extensions
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
    Union,
)

from browspi.services.dom.service import (
    DomService,
)  # Assuming this will exist or be moved
from browspi.services.ui_element_handler.service import (
    UiElementHandler,
)
from browspi.services.views import (
    DOMElementNode,
    SelectorMap,
)  # Assuming this will exist or be moved
from browspi.utils import time_execution_async  # Assuming this will exist or be moved

logger = logging.getLogger(__name__)

# --- Constants for Browser Configuration ---
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


class BrowserConfig(
    BrowserLaunchPersistentContextArgs, BrowserLaunchArgs, BrowserNewContextArgs
):
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        revalidate_instances="always",
        from_attributes=True,
        validate_by_name=True,  # Changed from validate_by_name to populate_by_name
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
    ) -> BrowserLaunchPersistentContextArgs:  # Adjusted return type
        return BrowserLaunchPersistentContextArgs(
            **self.model_dump(exclude={"args"}), args=self.get_args()
        )

    def kwargs_for_new_context(self) -> BrowserNewContextArgs:  # Adjusted return type
        return BrowserNewContextArgs(**self.model_dump(exclude={"args"}))

    def kwargs_for_launch(self) -> BrowserLaunchArgs:  # Adjusted return type
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


DEFAULT_BROWSER_PROFILE = BrowserConfig()


# --- WebAutomator Views (Simplified) ---
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
    element_tree: DOMElementNode  # This type comes from browspi.services.views
    selector_map: SelectorMap  # This type comes from browspi.services.views
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


class WebNavigator(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, frozen=False)
    browser_profile: BrowserConfig = Field(default_factory=BrowserConfig)
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

    async def start(self) -> "WebNavigator":
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
                    "user_data_dir must be set in BrowserConfig for persistent context and was not found in launch kwargs."
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
            f"WebNavigator started. Using {'persistent context at ' + str(self.browser_profile.user_data_dir) if self.browser_profile.user_data_dir and self.browser_profile.executable_path else 'new context'}."
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
        logger.info("WebNavigator stopped.")

    async def __aenter__(self) -> "WebNavigator":
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
        IGNORED_URL_PATTERNS = {
            "analytics",
            "tracking",
            "telemetry",
            "beacon",
            "metrics",
            "doubleclick",
            "adsystem",
            "adserver",
            "advertising",
            "facebook.com/plugins",
            "platform.twitter",
            "linkedin.com/embed",
            "livechat",
            "zendesk",
            "intercom",
            "crisp.chat",
            "hotjar",
            "push-notifications",
            "onesignal",
            "pushwoosh",
            "heartbeat",
            "ping",
            "alive",
            "webrtc",
            "rtmp://",
            "wss://",
            "cloudfront.net",
            "fastly.net",
        }

        async def on_request(request):
            if request.resource_type not in RELEVANT_RESOURCE_TYPES:
                return
            if request.resource_type in {
                "websocket",
                "media",
                "eventsource",
                "manifest",
                "other",
            }:
                return
            url = request.url.lower()
            if any(pattern in url for pattern in IGNORED_URL_PATTERNS):
                return
            if url.startswith(("data:", "blob:")):
                return
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

        async def on_response(response):
            request = response.request
            if request not in pending_requests:
                return
            content_type = response.headers.get("content-type", "").lower()
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
            if not any(ct in content_type for ct in RELEVANT_CONTENT_TYPES):
                pending_requests.remove(request)
                return
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB
                pending_requests.remove(request)
                return
            nonlocal last_activity
            pending_requests.remove(request)
            last_activity = asyncio.get_event_loop().time()

        page.on("request", on_request)
        page.on("response", on_response)
        now = asyncio.get_event_loop().time()
        try:
            start_time_loop = (
                asyncio.get_event_loop().time()
            )  # Renamed start_time to avoid conflict
            while True:
                await asyncio.sleep(0.1)
                now = asyncio.get_event_loop().time()
                if (
                    len(pending_requests) == 0
                    and (now - last_activity)
                    >= self.browser_profile.wait_for_network_idle_page_load_time
                ):
                    break
                if (
                    now - start_time_loop
                    > self.browser_profile.maximum_wait_page_load_time
                ):
                    logger.debug(
                        f"Network timeout after {self.browser_profile.maximum_wait_page_load_time}s with {len(pending_requests)} "
                        f"pending requests: {[r.url for r in pending_requests]}"
                    )
                    break
        finally:
            page.remove_listener("request", on_request)
            page.remove_listener("response", on_response)
        elapsed = now - start_time_loop  # Adjusted to use renamed variable
        if elapsed > 1:
            logger.debug(
                f"💤 Page network traffic calmed down after {elapsed:.2f} seconds"
            )

    async def _check_and_handle_navigation(self, page: Page) -> None:
        if not self._is_url_allowed(page.url):
            logger.warning(f"⛔️  Navigation to non-allowed URL detected: {page.url}")
            try:
                await self.go_back()  # Make sure go_back is defined or imported
            except Exception as e:
                logger.error(
                    f"⛔️  Failed to go back after detecting non-allowed URL: {str(e)}"
                )
            raise URLNotAllowedError(f"Navigation to non-allowed URL: {page.url}")

    async def _wait_for_page_and_frames_load(
        self, timeout_overwrite: float | None = None
    ):
        start_time_wait = time.time()  # Renamed start_time
        page = await self.get_current_page()
        try:
            await self._wait_for_stable_network()
            await self._check_and_handle_navigation(page)
        except URLNotAllowedError as e:
            raise e
        except Exception:
            logger.warning("⚠️  Page load failed, continuing...")
        elapsed = time.time() - start_time_wait  # Adjusted to use renamed variable
        remaining = max(
            (timeout_overwrite or self.browser_profile.minimum_wait_page_load_time)
            - elapsed,
            0,
        )
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
        tab_idx = (
            self.tabs.index(page) if page in self.tabs else -1
        )  # Added check for page in tabs
        if bytes_used is not None:
            logger.debug(
                f"➡️ Page navigation [{tab_idx}]{_log_pretty_url(page.url, 40)} used {bytes_used / 1024:.1f} KB in {elapsed:.2f}s, waiting +{remaining:.2f}s for all frames to finish"
            )
        else:
            logger.debug(
                f"➡️ Page navigation [{tab_idx}]{_log_pretty_url(page.url, 40)} took {elapsed:.2f}s, waiting +{remaining:.2f}s for all frames to finish"
            )
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def get_scroll_info(self, page: Page) -> tuple[int, int]:
        scroll_y = await page.evaluate("window.scrollY")
        viewport_height = await page.evaluate("window.innerHeight")
        total_height = await page.evaluate("document.documentElement.scrollHeight")
        pixels_above = scroll_y
        pixels_below = total_height - (scroll_y + viewport_height)
        return pixels_above, pixels_below

    @time_execution_async(
        "--remove_highlights"
    )  # This decorator needs to be defined or imported
    async def remove_highlights(self):
        page = await self.get_current_page()
        try:
            await page.evaluate(
                """
                try {
                    const container = document.getElementById('playwright-highlight-container');
                    if (container) container.remove();
                    const highlightedElements = document.querySelectorAll('[browser-user-highlight-id^="playwright-highlight-"]');
                    highlightedElements.forEach(el => el.removeAttribute('browser-user-highlight-id'));
                } catch (e) {
                    console.error('Failed to remove highlights:', e);
                }
                """
            )
        except Exception as e:
            logger.debug(
                f"⚠  Failed to remove highlights (this is usually ok): {type(e).__name__}: {e}"
            )

    async def get_state_summary(
        self, cache_clickable_elements_hashes: bool = True
    ) -> BrowserStateSummary:
        await self._wait_for_page_and_frames_load()
        updated_state = await self._get_updated_state()
        if cache_clickable_elements_hashes:
            if (
                self._cached_clickable_element_hashes
                and self._cached_clickable_element_hashes.url == updated_state.url
            ):
                updated_state_clickable_elements = (
                    UiElementHandler.get_clickable_elements(updated_state.element_tree)
                )
                for dom_element in updated_state_clickable_elements:
                    dom_element.is_new = (
                        UiElementHandler.hash_dom_element(dom_element)
                        not in self._cached_clickable_element_hashes.hashes
                    )
            self._cached_clickable_element_hashes = CachedClickableElementHashes(
                url=updated_state.url,
                hashes=UiElementHandler.get_clickable_elements_hashes(
                    updated_state.element_tree
                ),
            )
        assert updated_state
        self._cached_browser_state_summary = updated_state
        if self.browser_profile.cookies_file:
            asyncio.create_task(
                self.save_cookies()
            )  # Make sure save_cookies is defined
        return self._cached_browser_state_summary

    async def _get_updated_state(self, focus_element: int = -1) -> BrowserStateSummary:
        page = await self.get_current_page()
        try:
            await page.evaluate("1")
        except Exception as e:
            logger.debug(
                f"👋  Current page is no longer accessible: {type(e).__name__}: {e}"
            )
            raise BrowserError("Browser closed: no valid pages available")
        try:
            await self.remove_highlights()
            # Assuming DomService is correctly imported and initialized
            dom_service = DomService(page)
            content = await dom_service.get_clickable_elements(
                focus_element=focus_element,
                viewport_expansion=self.browser_profile.viewport_expansion,
                highlight_elements=self.browser_profile.highlight_elements,
            )
            tabs_info = await self.get_tabs_info()
            screenshot_b64 = await self.take_screenshot()
            pixels_above, pixels_below = await self.get_scroll_info(page)
            # Changed from self.browser_state_summary to a local variable
            browser_state_summary_val = BrowserStateSummary(
                element_tree=content.element_tree,
                selector_map=content.selector_map,
                url=page.url,
                title=await page.title(),
                tabs=tabs_info,
                screenshot=screenshot_b64,
                pixels_above=pixels_above,
                pixels_below=pixels_below,
            )
            return browser_state_summary_val  # Return the local variable
        except Exception as e:
            logger.error(f"❌  Failed to update state: {e}")
            if (
                hasattr(self, "_cached_browser_state_summary")
                and self._cached_browser_state_summary
            ):  # Check _cached_browser_state_summary
                return (
                    self._cached_browser_state_summary
                )  # Return _cached_browser_state_summary
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
        for i, page_item in enumerate(
            self.browser_context.pages
        ):  # Renamed page to page_item
            try:
                tabs_info.append(
                    TabInfo(page_id=i, url=page_item.url, title=await page_item.title())
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
            await self.get_current_page()  # Ensure a current page is set

    # Added missing methods referenced in the code
    async def go_back(self):
        page = await self.get_current_page()
        await page.go_back(wait_until="domcontentloaded")

    async def save_cookies(self):
        if self.browser_context and self.browser_profile.cookies_file:
            try:
                cookies = await self.browser_context.cookies()
                with open(self.browser_profile.cookies_file, "w") as f:
                    json.dump(cookies, f)  # Requires json import
                logger.info(f"Saved cookies to {self.browser_profile.cookies_file}")
            except Exception as e:
                logger.error(f"Failed to save cookies: {e}")

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

            # Recursively search for file input in node and its children
            def find_file_input_recursive(
                node: DOMElementNode, max_depth: int = 3, current_depth: int = 0
            ) -> DOMElementNode | None:
                if current_depth > max_depth or not isinstance(node, DOMElementNode):
                    return None

                # Check current element
                if is_file_input(node):
                    return node

                # Recursively check children
                if node.children and current_depth < max_depth:
                    for child in node.children:
                        if isinstance(child, DOMElementNode):
                            result = find_file_input_recursive(
                                child, max_depth, current_depth + 1
                            )
                            if result:
                                return result
                return None

            # Check if current element is a file input
            if is_file_input(candidate_element):
                return candidate_element

            # Check if it's a label pointing to a file input
            if (
                candidate_element.tag_name == "label"
                and candidate_element.attributes.get("for")
            ):
                input_id = candidate_element.attributes.get("for")
                root_element = get_root(candidate_element)

                target_input = find_element_by_id(root_element, input_id)
                if target_input and is_file_input(target_input):
                    return target_input

            # Recursively check children
            child_result = find_file_input_recursive(candidate_element)
            if child_result:
                return child_result

            # Check siblings
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
