# _(Not only)_ Coding standards

* **Python**

    [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) and [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) are in operation!

    Install `devel_requirements.txt` and run these commands (see below) in the root directory to check if your code comply with our coding standard:
    * `find . -type f -name "*.py" | xargs pylint`
    * `isort -rc . --diff --check-only`

* **Git**

    * [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) is in operation.
    * Remote branch names should follow those templates:
        * Personal branches: `<user name>/<your branch name>`
          These keep developer changes that are actively developed before merging into the master or one of develop branches.
        * Develop branches: `dev/<branch name e.g. r0.0.2>`
          These keep development code before release and merge into the master.


* **Merge requests**

    * If you want to commit to the master branch or one of develop branches **create a Merge Request**. Code-review and acceptance of at least one maintainer is mandatory.
