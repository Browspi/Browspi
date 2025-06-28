# src/browspi/use_cases/linkedin_apply.py

import csv
import os
from pathlib import Path

from pydantic import BaseModel
from PyPDF2 import PdfReader

from browspi.main import ActionManager, StepResult, WebNavigator

# --- Các thành phần dành riêng cho tác vụ LinkedIn ---

# Đường dẫn tới file CV
CV_PATH = Path.cwd() / "LiamHo_CV.pdf"


# Định nghĩa Pydantic model cho một Job
class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    location: str | None = None
    salary: str | None = None


# Tạo một ActionManager riêng cho tác vụ LinkedIn
linkedin_controller = ActionManager()


@linkedin_controller.action(
    "Save jobs to file - with a score how well it fits to my profile", param_model=Job
)
async def save_jobs(job: Job):
    """Lưu thông tin job vào file jobs.csv."""
    with open("jobs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([job.title, job.company, job.link, job.salary, job.location])
    return StepResult(extracted_content=f"Saved job '{job.title}' to file")


@linkedin_controller.action("Read my cv for context to fill forms")
async def read_cv():
    """Đọc nội dung từ file CV."""
    if not CV_PATH.exists():
        return StepResult(error=f"CV file not found at {CV_PATH}")
    try:
        pdf = PdfReader(CV_PATH)
        text = "".join(page.extract_text() or "" for page in pdf.pages)
        return StepResult(extracted_content=text, include_in_memory=True)
    except Exception as e:
        return StepResult(error=f"Failed to read CV: {e}")


@linkedin_controller.action(
    "Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element",
)
async def upload_cv(index: int, browser_session: WebNavigator):
    """Upload file CV lên một element trên trang web."""
    if not CV_PATH.exists():
        return StepResult(error=f"CV file not found at {CV_PATH}")

    path = str(CV_PATH.absolute())
    file_upload_dom_el = await browser_session.find_file_upload_element_by_index(index)

    if file_upload_dom_el is None:
        return StepResult(error=f"No file upload element found at index {index}")

    file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)
    if file_upload_el is None:
        return StepResult(
            error=f"Could not locate the file upload element handle at index {index}"
        )

    try:
        await file_upload_el.set_input_files(path)
        msg = f'Successfully uploaded file "{path}" to element with index {index}'
        return StepResult(extracted_content=msg)
    except Exception as e:
        return StepResult(error=f"Failed to upload file to index {index}: {str(e)}")


def get_linkedin_task_config():
    """
    Kiểm tra các yêu cầu, tạo prompt và trả về cấu hình cho tác vụ LinkedIn.

    Returns:
        tuple[str | None, ActionManager | None, str | None]:
        Một tuple chứa (task_prompt, controller, error_message).
    """
    username = os.getenv("LINKEDIN_USERNAME")
    password = os.getenv("LINKEDIN_PASSWORD")

    if not username or not password:
        error = "Error: LINKEDIN_USERNAME and LINKEDIN_PASSWORD must be set in your .env file."
        return None, None, error

    if not CV_PATH.exists():
        error = f"Error: CV file not found at the expected path: {CV_PATH}. Please place your CV there."
        return None, None, error

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
        * **Form Not Found / CV Upload Fails:** If the application form cannot be properly actioned, close the tab (if in a new one) and move to the next job.
        * **Stuck in Application:** If you find yourself stuck on a single job application for more than 5 distinct action steps, **skip this job**.
    """
    task = task.replace("{{LINKEDIN_USERNAME}}", username)
    task = task.replace("{{LINKEDIN_PASSWORD}}", password)

    return task, linkedin_controller, None
