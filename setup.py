from setuptools import find_packages, setup

setup(
    name="kgglm",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    description="""
    KGGLM: A Generative Language Model for Generalizable Knowledge Graph Representation in Recommendation
    """,
    author="Giacomo Balloccu, Ludovico Boratto, Gianni Fenu, Mirko Marras, Alessandro Soccol",
)
