import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cni-tlbx", # Replace with your own username
    version="1.2.0",
    author="Mario Senden",
    author_email="mario.senden@maastrichtuniversity.nl",
    description="computational neuroimaging toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ccnmaastricht/CNI_toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['numpy>=1.18.3',
                      'scipy>=1.4.1',
                      'opencv-python>=4.2.0.34'],
)
