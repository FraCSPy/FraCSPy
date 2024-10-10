# Contributing

Contributions are welcome and greatly appreciated!

The best way to get in touch with the core developers and maintainers of FraCSPy is to open new *Issues* directly from the GitHub repo.

## Welcomed contributions

### Bug reports

Report bugs at https://github.com/FraCSPy/fracspy/issues

If you are playing with the FraCSPy library and find a bug, please reporting it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### New algorithms

The best way to send feedback is to open an issue at
https://github.com/FraCSPy/fracspy/issues
with tag *enhancement*.

If you are proposing to include a new algorithm or a new feature  for an existing algorithm:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Fix issues
There is always a backlog of issues that need to be dealt with.
Look through the GitHub Issues for feature requests or bugfixes.

### Add examples or improve documentation
Writing new algorithms is not the only way to get involved and contribute. Create examples with existing algorithms as well as improving their documentation is as important as developing new algorithms and very much encouraged.


## Step-by-step instructions for contributing

Ready to contribute?

1. Follow all installation instructions in the [Step-by-step installation for developers](https://fracspy.github.io/FraCSPy/installation.html#step-by-step-installation-for-developers) section of the documentation.

2. Create a branch for local development, usually starting from the main branch:
    ```
    git checkout -b name-of-your-branch dev
    ```
    Now you can make your changes locally.

3. When you're done making changes, check that the both old and new tests pass successfully:
    ```
    make tests
    ```

4. Update the docs
   ```
   make docupdate
   ```

5. Commit your changes and push your branch to GitHub:
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.
    We recommend using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
    to format your commit messages, but this is not enforced.

6. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should ideally include some tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
3. Ensure that the updated code passes all tests.