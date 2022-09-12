import setuptools

setuptools.setup(
    name="hera_snapshot_imaging",
    version="0.0.0",
    author="Zachary Martinot",
    author_email="zmarti@sas.upenn.edu",
    description="A simple tool for producing snapshot images of HERA data.",
    packages=setuptools.find_packages(),
    python_requires='>=3.7'
)
