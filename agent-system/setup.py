from setuptools import setup, find_packages

setup(
    name="agent-system",
    version="1.0.0",
    description="Advanced modular agent system",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.11",
)
