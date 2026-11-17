"""Entry point for the frozen (PyInstaller) app."""
import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    from sc_analysis.gui.main_window import main
    sys.exit(main())
