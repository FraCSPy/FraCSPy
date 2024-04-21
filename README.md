# PyFrac

A single toolbox of python procedures for the full microseismic pipeline from modelling to post-event analysis.
This library is a single location leveraging the excellent work of other scientists (/software developers) and adapts them for the specific use case of microseismic monitoring.

Some functionalities include:

- modelling script generation (for accompanying [SOFI3D](https://docs.csc.fi/apps/sofi3d/))
- event imaging
- moment tensor inversion

Some python libraries that are heavily utilised include:

- pylops

## Requirements

Installation requires [Conda](https://conda.io) package manager, e.g. one can use [miniforge](https://github.com/conda-forge/miniforge).

## Install

Create a new conda env with all the required packages:

On Linux, from terminal:

```bash
./install_pyfrac.sh
```

This will also install xai_ssd in developer mode so you can change the source code and it will be updated automatically within the conda env.

Then activate the environment:

```bash
source activate pyfrac
```

On Windows, using miniforge prompt:

```cmd
install_pyfrac.bat
```

Now you are ready to use the package.

## Uninstall

If you need to add/change packages:

```bash
conda deactivate
conda remove -n pyfrac -all
```

## Documentation

The latest stable documentation based on [Sphinx](https://www.sphinx-doc.org) is available online at: <>

One can also build the documentation locally:

```bash
cd docs
make html
```

If you want to rebuild the documentation:

```bash
make clean
make html
```

After a successful build, one can serve the documentation website locally:

```bash
cd build/html
python -m http.server
```

and open in browser: <http://localhost:8000>

**Note:** check the exact port number in the output
