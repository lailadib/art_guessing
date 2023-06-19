from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='art guessing',
      version="0.0.2",
      description="art guessing",
      license="MIT",
      author="art guessing team & Le Wagon",
      author_email="contact@lewagon.org",
      #url="https://github.com/ TO BE DEFINED",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
