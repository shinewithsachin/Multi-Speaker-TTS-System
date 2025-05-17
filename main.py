import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
from ui.app import main

if __name__ == "__main__":
    main()