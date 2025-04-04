from setuptools import find_packages, setup

setup(
    name="aiworker-llmada",
    version="0.1.0",
    author="zhaoxuefeng",
    author_email="823042332@qq.com",
    description="负责大模型的方便调度",
    url="",
    packages=find_packages(),
    install_requires=[
        "pyyaml~=6.0.2",
    ],
    python_requires=">=3.10",
)
