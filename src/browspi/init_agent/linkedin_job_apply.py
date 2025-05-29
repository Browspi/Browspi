import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from PyPDF2 import PdfReader

from browspi.main import (
    ActionResult,
    Agent,
    AgentSettings,
    BrowserSession,
    Controller,
    load_dotenv,
    logging,
)
from browspi.services.browser.service import DEFAULT_BROWSER_PROFILE

load_dotenv()
logger = logging.getLogger(__name__)


job_controller = Controller()

# NOTE: This is the path to your cv file
CV = Path.cwd() / "LiamHo_CV.pdf"

if not CV.exists():
    raise FileNotFoundError(
        f"You need to set the path to your cv file in the CV variable. CV file not found at {CV}"
    )


class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    location: str | None = None
    salary: str | None = None


@job_controller.action(
    "Save jobs to file - with a score how well it fits to my profile", param_model=Job
)
async def save_jobs(job: Job):  # Changed to async def
    with open("jobs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([job.title, job.company, job.link, job.salary, job.location])
    # Return ActionResult for consistency, though the framework can handle string returns too
    return ActionResult(extracted_content="Saved job to file")


@job_controller.action("Read jobs from file")
async def read_jobs():  # Changed to async def
    with open("jobs.csv") as f:
        content = f.read()
    return ActionResult(extracted_content=content, include_in_memory=True)


@job_controller.action("Read my cv for context to fill forms")
async def read_cv():  # Changed to async def
    pdf = PdfReader(CV)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    logger.info(f"Read cv with {len(text)} characters")
    return ActionResult(extracted_content=text, include_in_memory=True)


@job_controller.action(
    "Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element",
)
async def upload_cv(index: int, browser_session: BrowserSession):
    path = str(CV.absolute())
    file_upload_dom_el = await browser_session.find_file_upload_element_by_index(index)

    if file_upload_dom_el is None:
        logger.info(f"No file upload element found at index {index}")
        return ActionResult(error=f"No file upload element found at index {index}")

    file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)

    if file_upload_el is None:
        logger.info(f"No file upload element found at index {index}")
        return ActionResult(error=f"No file upload element found at index {index}")

    try:
        await file_upload_el.set_input_files(path)
        msg = f'Successfully uploaded file "{path}" to index {index}'
        logger.info(msg)
        return ActionResult(extracted_content=msg)
    except Exception as e:
        logger.debug(f"Error in set_input_files: {str(e)}")
        return ActionResult(error=f"Failed to upload file to index {index}")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    ground_task = """
        You are an AI agent tasked with finding and applying to jobs that match my profile.
        Your goal is to find jobs that match my skills and experience, and apply to them using the provided CV.
        1. Read my cv with using read_cv function. And remember it to fill the job application form.
        2. Search for jobs that match my profile using the provided job search engine. Try to hit Enter key on the search input field to submit the search.
        3. Check carefully that you are in the job listings. The previous search must be success to come there. Filter the jobs by hitting the button "Easy Apply"
        4. Apply to the job using the provided cv by uploading it to the job application form.
        5. If it open a Chrome new tab, switch to the new tab and find the job application form.
            5.1. If the job application form is not found, try to find the file upload element by index and upload the cv.
            5.2. If the job application form is not found, then skip the job and move to the next one.
            5.3. If the job application form is found, then fill in the form with my cv and submit it.
            5.4. If no job application form is found, switch back to the previous Chrome tab and continue searching for jobs.
        6. If the job application form is not found, then skip the job and move to the next one.
        7. If the job application form is found, then fill in the form with my cv and submit it.
        8. If the job application form is not found, then skip the job and move to the next one.
        9. If the job application form is found, then fill in the form with my cv and submit it.
    Hint: 
    1. If you encounter a security check or captcha, please wait for 15 seconds from me to solve it before proceeding. Or go back to the previous tab and continue searching for jobs.
    2. If you stuck at step applying to the job for more than 5 steps, then try to move to the next job.
    """

    task = (
        ground_task
        + """
        Please find those jobs in this Linkedin website: https://www.linkedin.com/jobs
        If login is required. Login with the credentials provided in the environment variables:
        Linkedin username: {{LINKEDIN_USERNAME}}
        Linkedin password: {{LINKEDIN_PASSWORD}}
        After logging in, search for jobs that match my profile.
        Hint: 
        1. If you cannot find the search submission form, try to hit Enter key on the search input field to submit the search.
        2. If you stuck at choosing the options in the application form, try to click on the option element to select it.
        3. If you encounter a select element, try to click on the select element to open the dropdown and then click on the option you want to select.
        4. If you stuck at submitting the application form, try to recheck the form for any missing fields or errors.
    """
    )

    task = task.replace("{{LINKEDIN_USERNAME}}", os.getenv("LINKEDIN_USERNAME"))
    task = task.replace("{{LINKEDIN_PASSWORD}}", os.getenv("LINKEDIN_PASSWORD"))

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
        controller=job_controller,
    )
    try:
        print(f"🚀 Starting agent task: {task}")
        history = await agent.run(max_steps=50)
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
