from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="creditlens",
    version="0.1.0",
    author="Rohan Khiani",
    author_email="rohan.s.khiani.vora@gmail.com",
    description="An explainable credit-scoring reference: every decision has an attribution that sums exactly to the served probability, and the fairness audit reports intervals and says INCONCLUSIVE when it should.",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ROROK20/CreditLens",
    packages=find_packages(exclude=["tests*"]),
    # >=3.12: forced by the exact pins below — shap==0.52.0 requires >=3.12
    # (pandas/numpy/scipy/scikit-learn require >=3.11). The documented demo numbers
    # reproduce on CPython 3.12.5; suite re-run 2026-11-05 on the same
    # interpreter: 526 passed, 0 skipped.
    python_requires=">=3.12",
    # EXACT pins, kept in lockstep with requirements.txt (see the rationale
    # there): the seeded pipeline's documented numbers reproduce only under a
    # fixed dependency set.
    install_requires=[
        "pandas==3.0.3", "numpy==2.4.6", "scipy==1.17.1",
        "scikit-learn==1.8.0", "xgboost==3.2.0", "joblib==1.5.3",
        "fastapi==0.136.3", "uvicorn==0.49.0",
        "pydantic==2.13.4", "python-dotenv==1.2.2",
    ],
    extras_require={
        "xai": ["shap==0.52.0"],
        "dev": ["pytest==9.0.3", "pytest-cov==7.1.0", "httpx==0.28.1"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
