"""TraceView entry point: python app.py [image.pdf | work.trproj]"""
import sys

from traceview.gui.main_window import main

if __name__ == "__main__":
    sys.exit(main())
