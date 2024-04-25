from setuptools import setup, find_packages

setup(
    name="lumibot_tools",
    version="0.1.6",
    packages=find_packages(),
    install_requires=[
        'lumibot',  # Añade lumibot como una dependencia
        'aiogram'
    ],
    scripts=['bin/run.py'],
    # Más metadata:
    author="Rubén Cancho",
    author_email="canchete@gmail.com",
    description="Herramientas útiles para Lumibot, incluyendo mensajería y ejecución de estrategias.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Canx/lumibot_tools",
)
