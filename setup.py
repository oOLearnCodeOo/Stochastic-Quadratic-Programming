from setuptools import setup, find_packages

setup(
    name="stochastic-sqp-optimizer",
    version="0.1.0",
    description="A Python implementation of Stochastic SQP for optimization problems.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),  # 自动查找项目中的所有包
    install_requires=[
        "torch>=1.9.0"
    ],
    python_requires=">=3.7",  # 指定支持的 Python 版本
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)