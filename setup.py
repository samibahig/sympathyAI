from setuptools import setup, find_packages

setup(
    name="sympathyai",
    version="0.1.0",
    description="Structure-aware evaluation and alignment for LLM reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SympathyAI Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "llm": ["openai>=1.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
