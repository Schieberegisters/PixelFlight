import pytest
import sys
import os
from pathlib import Path

def run_all_tests():
    """
    Discovers and executes all test scripts inside the tests/ directory.
    """
    # 1. Get the absolute path of the project root
    root_dir = Path(__file__).resolve().parent
    
    # 2. Add the root directory to sys.path
    # This ensures that 'Static' and 'Dynamic' modules are findable
    # regardless of where the script is called from.
    sys.path.append(str(root_dir))
    
    print("=" * 60)
    print(f"🚀 Starting Sign Language Project Test Suite")
    print(f"📍 Root Directory: {root_dir}")
    print("=" * 60)

    # 3. Configure Pytest arguments
    # -v: Verbose output
    # tests: The directory to search for tests
    # --disable-warnings: Keeps the output clean from library deprecation notices
    args = [
        "-v",
        "tests",
        "--disable-warnings",
        "-p", "no:cacheprovider"
    ]

    # 4. Run Pytest
    exit_code = pytest.main(args)

    # 5. Exit with the appropriate code (0 for success, 1 for failure)
    sys.exit(exit_code)

if __name__ == "__main__":
    run_all_tests()