# src/browspi/ui/browser_utils.py

import os


def get_persistent_profile_path() -> str | None:
    """
    Lấy đường dẫn đến thư mục profile trình duyệt từ biến môi trường
    'PERSISTENT_PROFILE_PATH', giống hệt như trong main.py.
    """
    path = os.getenv("PERSISTENT_PROFILE_PATH")
    if path:
        print(f"Found PERSISTENT_PROFILE_PATH: {path}")
    else:
        print("Warning: PERSISTENT_PROFILE_PATH environment variable not set.")
    return path


def get_chrome_executable_path() -> str | None:
    """
    Lấy đường dẫn đến file thực thi của Chrome từ biến môi trường
    'CHROME_EXE_PATH', giống hệt như trong main.py.
    """
    path = os.getenv("CHROME_EXE_PATH")
    if path:
        print(f"Found CHROME_EXE_PATH: {path}")
    else:
        print("Warning: CHROME_EXE_PATH environment variable not set.")
    return path
