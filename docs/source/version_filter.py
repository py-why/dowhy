import os

is_ci = os.environ.get("CI", False)


def create_version_filter(include: str):
    # Create a negative-lookahead regexp to exclude the version from being generated
    def exclude(dir):
        return "(?!" + dir.replace(".", "\.") + ")"

    versions = os.listdir("../../dowhy-docs")
    versions.remove("index.html")
    versions.remove(".nojekyll")
    versions.remove("main")
    result = "^".join(list(map(exclude, versions))) + include + "$"
    return result


def create_branch_filter():
    result = "main"
    if is_ci == "true":
        # build the currently checked-out branch, which should be the only branch in GH Actions
        result = "^.*$"

    return result
