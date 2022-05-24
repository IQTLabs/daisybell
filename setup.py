from setuptools import setup, find_packages

with open("requirements.txt") as fd:
    install_requires = fd.read().splitlines()

setup(
    name="daisybell",
    version="0.3.2",
    description="Scan AI models for problems",
    long_description=open("README.rst").read(),
    keywords="machine_learning artificial_intelligence",
    author="JJ Ben-Joseph",
    author_email="jbenjoseph@iqt.org",
    python_requires=">=3.8",
    url="https://github.com/IQTLabs/daisybell",
    license="Apache",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=install_requires,
    tests_require=["pytest", "pre-commit"],
    entry_points={"console_scripts": ["daisybell = daisybell.__main__:main"]},
)
