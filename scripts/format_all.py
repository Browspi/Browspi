import subprocess


def main():
    print("🧹 Removing unused imports with autoflake...")
    subprocess.run(
        "poetry run autoflake --remove-all-unused-imports --in-place --recursive .",
        shell=True,
        check=True,
    )

    print("🎨 Formatting code and fixing long lines with black...")
    subprocess.run("poetry run black . --line-length 100", shell=True, check=True)

    print("🎨 Sorting and formatting imports with ruff...")
    subprocess.run("poetry run ruff format .", shell=True, check=True)

    print("🔍 Final lint check with ruff...")
    subprocess.run("poetry run ruff check . --fix", shell=True, check=False)

    print("✅ All formatting steps completed.")
