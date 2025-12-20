#!/usr/bin/env python
"""Main entry point to run the hotel booking prediction pipeline."""
import sys
import os

# Add the inner Hotel Booking package to Python path
inner_pkg = os.path.join(os.getcwd(), 'Hotel Booking')
if inner_pkg not in sys.path:
    sys.path.insert(0, inner_pkg)

# Now import from the inner package
from pipeline.run_pipeline import run

if __name__ == '__main__':
    run()
