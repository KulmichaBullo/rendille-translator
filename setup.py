from setuptools import setup, find_packages
setup(
    name="rendille-translation",
    version="0.1.0",
    description="Neural Machine Translation for Rendille→English",
    author="Meelilabs Toola Group",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "sacrebleu>=2.3.0",
        "sentencepiece",
        "peft",
        "accelerate",
    ],
    extras_require={
        "dev": ["flake8", "black", "jupyter"],
        "api": ["flask", "ctranslate2"],
    },
)
