from setuptools import setup, find_packages

setup(
    name="mlitfastapi",
    version="0.1.0",
    packages=find_packages(include=["mlitfastapi", "mlitfastapi.*"]),
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "pandas",
        "scikit-learn",
        "joblib",
        "xgboost",
        "python-multipart",
    ],
    description="API de optimización de inventario para PYMES",
    author="Federico/Paulino",
    python_requires='>=3.7',
)
