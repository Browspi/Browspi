import logging  # Added for DOMElementNode's clickable_elements_to_string
import time  # for time_execution_async/sync
from dataclasses import dataclass
from functools import wraps  # wraps for time_execution_async/sync
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # Added List, Dict, Any

# Avoid circular import issues
if TYPE_CHECKING:
    from .dom_definitions import DOMElementNode  # Self-reference for parent type

logger = logging.getLogger(__name__)  # Added


# Copied from your main.py - ensure these are identical or adapt as needed
def time_execution_sync(additional_text: str = "") -> callable:
    def decorator(func: callable) -> callable:
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


def time_execution_async(additional_text: str = "") -> callable:
    def decorator(func: callable) -> callable:
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


# Simplified ViewportInfo, CoordinateSet, HashedDomElement for now
# to avoid pulling in the entire history_tree_processor
@dataclass(frozen=True)
class ViewportInfo:
    width: int
    height: int


CoordinateSet = Optional[Any]  # Placeholder type
HashedDomElement = Optional[Any]  # Placeholder type


@dataclass(frozen=False)
class DOMBaseNode:
    is_visible: bool
    parent: Optional["DOMElementNode"]

    def __json__(self) -> dict:
        raise NotImplementedError("DOMBaseNode is an abstract class")


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
    text: str
    type: str = "TEXT_NODE"

    def has_parent_with_highlight_index(self) -> bool:
        current = self.parent
        while current is not None:
            if current.highlight_index is not None:
                return True
            current = current.parent
        return False

    # These methods were in browser-use/dom/views.py, keep them if DomService or other parts might use them
    def is_parent_in_viewport(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.is_in_viewport

    def is_parent_top_element(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.is_top_element

    def __json__(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
        }


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
    tag_name: str
    xpath: str
    attributes: Dict[str, str]
    children: List[DOMBaseNode]  # Use List from typing
    is_interactive: bool = False
    is_top_element: bool = False
    is_in_viewport: bool = False
    shadow_root: bool = False
    highlight_index: int | None = None
    viewport_coordinates: CoordinateSet | None = None
    page_coordinates: CoordinateSet | None = None
    viewport_info: ViewportInfo | None = None  # Uses the local ViewportInfo
    is_new: bool | None = None  # Kept from your main.py for compatibility

    # @cached_property
    # def hash(self) -> HashedDomElement:
    #     # Commenting out for now to avoid HistoryTreeProcessor dependency
    #     # from browspi.dom.history_tree_processor.service import HistoryTreeProcessor
    #     # return HistoryTreeProcessor._hash_dom_element(self)
    #     logger.warning("DOMElementNode.hash is currently not implemented in this version.")
    #     return None

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        # This is the implementation from browser-use/dom/views.py
        text_parts = []

        def collect_text(node: DOMBaseNode, current_depth: int) -> None:
            if max_depth != -1 and current_depth > max_depth:
                return
            if (
                isinstance(node, DOMElementNode)
                and node != self
                and node.highlight_index is not None
            ):
                return
            if isinstance(node, DOMTextNode):
                text_parts.append(node.text)
            elif isinstance(node, DOMElementNode):
                for child in node.children:
                    collect_text(child, current_depth + 1)

        collect_text(self, 0)
        return "\n".join(text_parts).strip()

    @time_execution_sync("--clickable_elements_to_string")
    def clickable_elements_to_string(
        self, include_attributes: list[str] | None = None
    ) -> str:
        # This is the full implementation from browser-use/dom/views.py
        formatted_text = []

        def process_node(node: DOMBaseNode, depth: int) -> None:
            next_depth = int(depth)
            depth_str = depth * "\t"
            if isinstance(node, DOMElementNode):
                if node.highlight_index is not None:
                    next_depth += 1
                    text = node.get_all_text_till_next_clickable_element()
                    attributes_html_str = ""
                    if include_attributes:
                        attributes_to_include = {
                            key: str(value)
                            for key, value in node.attributes.items()
                            if key in include_attributes
                        }
                        if node.tag_name == attributes_to_include.get("role"):
                            del attributes_to_include["role"]
                        if (
                            attributes_to_include.get("aria-label")
                            and attributes_to_include.get("aria-label", "").strip()
                            == text.strip()
                        ):
                            del attributes_to_include["aria-label"]
                        if (
                            attributes_to_include.get("placeholder")
                            and attributes_to_include.get("placeholder", "").strip()
                            == text.strip()
                        ):
                            del attributes_to_include["placeholder"]
                        if attributes_to_include:
                            attributes_html_str = " ".join(
                                f"{key}='{value}'"
                                for key, value in attributes_to_include.items()
                            )

                    if node.is_new:  # check if this field exists, it's from your main.py's original DOMElementNode
                        highlight_indicator = f"*[{node.highlight_index}]*"
                    else:
                        highlight_indicator = f"[{node.highlight_index}]"
                    line = f"{depth_str}{highlight_indicator}<{node.tag_name}"
                    if attributes_html_str:
                        line += f" {attributes_html_str}"
                    if text:
                        if not attributes_html_str:
                            line += " "
                        line += f">{text}"
                    elif not attributes_html_str:
                        line += " "
                    line += " />"
                    formatted_text.append(line)
                for child in node.children:
                    process_node(child, next_depth)
            elif isinstance(node, DOMTextNode):
                if (
                    not node.has_parent_with_highlight_index()
                    and node.parent
                    and node.parent.is_visible
                    and node.parent.is_top_element
                ):
                    formatted_text.append(f"{depth_str}{node.text}")

        process_node(self, 0)
        return "\n".join(formatted_text)

    # __json__ and __repr__ from browser-use/dom/views.py can be added here if needed by other parts
    # For brevity, I'm omitting them but you can copy them from the file if your agent uses them.


SelectorMap = Dict[int, DOMElementNode]  # Use Dict from typing


@dataclass
class DOMState:
    element_tree: DOMElementNode
    selector_map: SelectorMap
