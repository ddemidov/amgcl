from subprocess import Popen, PIPE

#----------------------------------------------------------------------------
# Get version string from git
#
# Author: Douglas Creager <dcreager@dcreager.net>
# http://dcreager.net/2010/02/10/setuptools-git-version-numbers/
#
# PEP 386 adaptation from
# https://gist.github.com/ilogue/2567778/f6661ea2c12c070851b2dfb4da8840a6641914bc
#----------------------------------------------------------------------------
def call_git_describe(abbrev=4):
    try:
        p = Popen(['git', 'describe', '--abbrev=%d' % abbrev],
                  stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        return line.strip().decode('utf8')

    except:
        return None


def read_release_version():
    try:
        f = open("RELEASE-VERSION", "r")

        try:
            version = f.readlines()[0]
            return version.strip()

        finally:
            f.close()

    except:
        return None


def write_release_version(version):
    f = open("RELEASE-VERSION", "w")
    f.write("%s\n" % version)
    f.close()


def pep386adapt(version):
    # adapt git-describe version to be in line with PEP 386
    parts = version.split('-')
    if len(parts) > 1:
        parts[-2] = 'post'+parts[-2]
        version = '.'.join(parts[:-1])
    return version


def git_version(abbrev=4):
    # Read in the version that's currently in RELEASE-VERSION.

    release_version = read_release_version()

    # First try to get the current version using "git describe".

    version = call_git_describe(abbrev)

    # If that doesn't work, fall back on the value that's in
    # RELEASE-VERSION.

    if version is None:
        version = release_version
    else:
        #adapt to PEP 386 compatible versioning scheme
        version = pep386adapt(version)


    # If we still don't have anything, that's an error.

    if version is None:
        raise ValueError("Cannot find the version number!")

    # If the current version is different from what's in the
    # RELEASE-VERSION file, update the file to be current.

    if version != release_version:
        write_release_version(version)

    # Finally, return the current version.

    return version
