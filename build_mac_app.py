#!/usr/bin/env python3
"""
Build script to create a Mac App bundle for WorkSmart AI Coach
Uses py2app to create a standalone .app
"""

import os
import sys
from setuptools import setup

# This script creates a Mac .app bundle
APP = ['worksmart_ai_coach/launchers/enhanced.py']

DATA_FILES = [
    ('', ['.env.template']),
    ('', ['README.md', 'LICENSE']),
]

OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'LSUIElement': True,  # Hide from dock initially
        'CFBundleName': 'WorkSmart AI Coach',
        'CFBundleDisplayName': 'WorkSmart AI Coach',
        'CFBundleIdentifier': 'io.crossover.worksmart-ai-coach',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2025 CrossOver. All rights reserved.',
        'LSBackgroundOnly': False,
        'LSApplicationCategoryType': 'public.app-category.productivity',
    },
    'packages': ['worksmart_ai_coach'],
    'iconfile': 'icon.icns',  # You'll need to create this
    'resources': ['.env.template', 'README.md'],
    'includes': [
        'anthropic',
        'pynput', 
        'psutil',
        'click',
        'numpy',
        'aiofiles',
        'dotenv'
    ],
    'excludes': [
        'tkinter',
        'unittest',
        'distutils',
    ],
}

if __name__ == '__main__':
    setup(
        app=APP,
        data_files=DATA_FILES,
        options={'py2app': OPTIONS},
        setup_requires=['py2app'],
    )