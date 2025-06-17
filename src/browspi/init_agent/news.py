import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browspi.main import (
    ActionManager,
    AutomationConfig,
    WebAutomator,
    load_dotenv,
    logging,
)
from browspi.services.browser.service import DEFAULT_BROWSER_PROFILE

load_dotenv()
logger = logging.getLogger(__name__)


new_controller = ActionManager()


class New(BaseModel):
    title: str
    author: str
    link: str
    date_published: str
    summary: str = None


@new_controller.action(
    "Save news into file news_data.csv",
    param_model=New,
)
async def save_news_data(new: New):
    """Save the news data to a dated CSV file in /output/news/YYYY-MM-DD.csv within the project root."""
    try:
        # Resolve the project root using the current file's location
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "output" / "news"
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the directory if it doesn't exist

        # Generate the filename based on today's date
        today_str = datetime.today().strftime("%Y-%m-%d")
        file_path = output_dir / f"{today_str}.csv"

        # Open the file in append mode and write data
        with open(file_path, "a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(
                    ["Title", "Author", "Link", "Date Published", "Summary"]
                )
            writer.writerow(
                [new.title, new.author, new.link, new.date_published, new.summary]
            )

        return "Saved news to file"
    except Exception as e:
        print(f"Error saving news data: {e}")
        logger.error("Error saving news data", exc_info=True)


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    task = """
    Research the latest developments regarding {topic} from at least 5 different reputable news sources.

    For each source:
    1. Navigate to their search function and find articles about the topic from the past week
    2. Extract the headline, publication date, and author, and save the link to the article with save_news_data function.
    3. Summarize the key points in 2-3 sentences

    After gathering information, synthesize the findings into a comprehensive summary that notes any differences in reporting or perspective between the sources.

    Tips:
    - If you encounter a captcha please wait for 10 seconds before retrying.
    - Try to search on google first to find the latest articles.
    - Don't treat a google preview as a full article, always click through to the original source.
    
    Ensure that the summary is concise and highlights the most significant developments.
    """

    topic = "Covid-19 in Vietnam"
    task = task.format(topic=topic)

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
    current_as = AutomationConfig(
        use_vision=True,
        max_actions_per_step=3,
        tool_calling_method="tools",
        page_extraction_llm=llm,
        save_conversation_path=os.path.join(
            "conversations",
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_llm_conversation_history.json",
        ),
    )
    agent = WebAutomator(
        task=task,
        llm=llm,
        agent_settings=current_as,
        browser_profile=current_bp,
        controller=new_controller,
    )
    try:
        print(f"🚀 Starting agent task: {task}")
        history = await agent.start_task(max_steps=50)
        print("\n--- WebAutomator Run History ---")
        if history.history:
            for i, hist_item in enumerate(history.history):
                print(
                    f"\n--- History Step {i + 1} (WebAutomator Step {hist_item.metadata.get('step', 'N/A') if hist_item.metadata else 'N/A'}) ---"
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
            print(
                "\n WebAutomator did not complete successfully or produce a final result."
            )
    except Exception as e:
        print(f"Error during agent execution: {e}")
        logger.error("Main execution error", exc_info=True)
    finally:
        await agent.close()
        print("Closing browser session...")


def __main__():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())


if __name__ == "__main__":
    __main__()
