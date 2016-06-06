try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='ilid-preprocessing',
    version='0.1',
    packages=['audio', 'graphic', 'output', 'util'],
    url='https://github.com/twerkmeister/iLID',
    license='',
    author='Thomas Werkmeister',
    author_email='thomas@student.hpi.de',
    description='Spark Preprocessing Pipeline',
    requires=['numpy', 'scipy', 'cv2']
)