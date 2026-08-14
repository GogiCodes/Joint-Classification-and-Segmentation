"""
Setup script for MultiTask-ResNet18-Vision
"""

from setuptools import setup, find_packages

setup(
    name='multitask-resnet18-vision',
    version='1.0.0',
    description='Multi-Task Learning for Image Classification and Semantic Segmentation',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.20.0',
        'pillow>=9.0.0',
        'tensorboard>=2.10.0',
        'scikit-learn>=1.0.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
