#!/usr/bin/env python3
"""
Entry point for training the Appraise-ATN model.
Wraps the core training script in appraise_plm/train.py.
"""

import os
import sys
import subprocess

if __name__ == '__main__':
    # Construct the path to the actual training script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, 'appraise_plm', 'train.py')
    
    # Run the script with all passed arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
