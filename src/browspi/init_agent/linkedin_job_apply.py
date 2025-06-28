import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader

from browspi.main import (
    ActionManager,
    AutomationConfig,
    StepResult,
    WebAutomator,
    WebNavigator,
    load_dotenv,
    logging,
)
from browspi.services.browser.service import DEFAULT_BROWSER_PROFILE

load_dotenv()
logger = logging.getLogger(__name__)


job_controller = ActionManager()

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
    # Return StepResult for consistency, though the framework can handle string returns too
    return StepResult(extracted_content="Saved job to file")


@job_controller.action("Read jobs from file")
async def read_jobs():  # Changed to async def
    with open("jobs.csv") as f:
        content = f.read()
    return StepResult(extracted_content=content, include_in_memory=True)


@job_controller.action("Read my cv for context to fill forms")
async def read_cv():  # Changed to async def
    pdf = PdfReader(CV)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    # logger.info(f"Read cv with {len(text)} characters")
    logger.info(f"{text}")
    return StepResult(extracted_content=text, include_in_memory=True)


@job_controller.action(
    "Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element",
)
async def upload_cv(index: int, browser_session: WebNavigator):
    path = str(CV.absolute())
    file_upload_dom_el = await browser_session.find_file_upload_element_by_index(index)

    if file_upload_dom_el is None:
        logger.info(f"No file upload element found at index {index}")
        return StepResult(error=f"No file upload element found at index {index}")

    file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)

    if file_upload_el is None:
        logger.info(f"No file upload element found at index {index}")
        return StepResult(error=f"No file upload element found at index {index}")

    try:
        await file_upload_el.set_input_files(path)
        msg = f'Successfully uploaded file "{path}" to index {index}'
        logger.info(msg)
        return StepResult(extracted_content=msg)
    except Exception as e:
        logger.debug(f"Error in set_input_files: {str(e)}")
        return StepResult(error=f"Failed to upload file to index {index}")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    task = """
    You are an AI Job Application Assistant. Your primary goal is to find suitable job openings on LinkedIn that match the skills and experience detailed in the provided CV, and then apply to those jobs.

    **Overall Workflow:**

    **Phase 1: Setup & Initial Search**
    1.  **Load Profile:** Begin by using the `read_cv` function to load the content of my CV. This content is "my profile" for job matching.
    2.  **Navigate to LinkedIn Jobs:** Go directly to `https://www.linkedin.com/jobs`.
    3.  **Login (if required):**
        * If a login page appears, use the following credentials (provided as environment variables):
            * Username: `{{LINKEDIN_USERNAME}}`
            * Password: `{{LINKEDIN_PASSWORD}}`
    4.  **Initial Job Search:**
        * Once logged in (or if no login was needed), search for jobs that align with the details extracted from my CV.
        * **Action Tip:** Hit enter to submit the search query after entering keywords in the search bar.

    **Phase 2: Job Filtering & Application (Loop per suitable job)**
    5.  **Filter for "Easy Apply":** On the job listings page, prioritize jobs with an "Easy Apply" option. Click the "Easy Apply" filter button if available.
    6.  **Identify Suitable Job & Initiate Application:**
        * Select a promising job from the filtered list.
        * **Crucial Check:** Before proceeding, verify if the "Apply" button links to an external website. If it does, **skip this job** and move to the next one.
        * If it's an internal LinkedIn "Easy Apply", proceed.
    7.  **Application Form Handling:**
        * **New Tab Management:** If the application process opens in a new Chrome tab, switch to this new tab.
        * **Locate Application Form:** Find the job application form on the page.
        * **Form Filling:** Use the information extracted from my CV (via `read_cv` in step 1) to fill out the application form fields (e.g., name, email, phone number, experience details).
        * **CV Upload:**
            * Locate the file upload element for the CV.
            * Use the `upload_cv` function with the appropriate element index. If the first attempt fails, try to identify the correct index for the upload element and call `upload_cv` again.
        * **Submission:** After filling the form and uploading the CV, submit the application.
    8.  **Handling Application Issues & Skipping:**
        * **Form Not Found / CV Upload Fails:** If, after reasonable attempts (e.g., trying different indices for `upload_cv`), the application form cannot be properly actioned (e.g., form not found, CV upload element persistently not working), then:
            * If you are in a new tab, close it and switch back to the main LinkedIn jobs tab.
            * **Skip this job** and move to the next suitable job in the listings.
        Don't upload_cv, as it already uploaded the CV on LinkedIn.
        * **Stuck in Application:** If you find yourself stuck on a single job application for more than 5 distinct action steps (e.g., repeated failed attempts to fill a field, click a button, or upload CV), **skip this job** and move to the next one.

    **General Guidelines & Action Tips:**

    * **Security Checks/CAPTCHAs:** If you encounter a security check or CAPTCHA:
        * First, use the `wait` action for 15 seconds to allow manual intervention.
        * If unresolved, and if you are on an application page, try to go back to the LinkedIn jobs tab and select a different job.
    * **Using CV Data:** When filling forms, actively recall and use the specific details (name, email, phone, work history, skills) obtained from the `read_cv` function.
    * **Dropdowns/Select Elements:** If you encounter a `<select>` dropdown element:
        1.  Click the `<select>` element itself to open the dropdown.
        2.  Then, click the desired `<option>` element within the opened dropdown.
    * **Form Submission Issues:** If an application submission fails, re-examine the form for any highlighted errors or missing required fields before attempting to submit again.
    * **Be Methodical:** Break down complex forms into smaller, logical steps.
    """

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
        controller=job_controller,
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
        print("Closing browser session...")
        await agent.close()


def __main__():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())


if __name__ == "__main__":
    __main__()
