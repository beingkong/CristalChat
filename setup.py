"""
Setup script for the Voice Chatbot package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="voice-chatbot",
    version="0.1.0",
    author="AI Assistant",
    author_email="example@example.com",
    description="A multi-module voice AI chat system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/voice-chatbot",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "sounddevice",
        "soundfile",
        "torch",
        "asyncio",
    ],
    entry_points={
        "console_scripts": [
            "voice-chatbot=voice_chatbot.main:main",
        ],
    },
    include_package_data=True,
)
