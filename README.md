# pyfrac

A single toolbox of python procedures for the full microseismic pipeline from modelling to post-event analysis.
This library is a single location leveraging the excellent work of other scientists (/software developers) and 
adapts them for the specific usecase of microseismic monitoring.

Some functionalities include:
- modelling script generation (for accompanying SOFI3D),
- standard signal processing,
- event imaging

Some python libraries that are heavily utilised include:
- pylops


# Setup

Create a new conda env with all the required packages:
    
    ./install_pyfrac.sh

This will also install xai_ssd in developer mode so you can change the source code and it will be 
updated automatically within the conda env. Then activate the environment:

    source activate pyfrac

Now you are ready to use the package. If you need to add/change packages:

    conda deactivate
    conda env remove -n pyfrac

