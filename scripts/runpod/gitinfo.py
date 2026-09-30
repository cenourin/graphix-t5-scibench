#!/usr/bin/env python
"""Commit of the checkout without the git binary (the a5 image has no git).

  python scripts/runpod/gitinfo.py [REPO]      # prints the full commit sha (exit 1 if unknown)

Order: a COMMIT file at the repo root (written when code is shipped as an archive), else
.git/HEAD -> the ref it names -> .git/refs/heads/... or .git/packed-refs. Plain file reads
only. Does not detect uncommitted changes (that needs git); the session records file hashes
of the scripts it runs instead.
"""
import os
import sys


def commit(repo="."):
    f = os.path.join(repo, "COMMIT")
    if os.path.isfile(f):
        return open(f).read().split()[0]
    git = os.path.join(repo, ".git")
    if os.path.isfile(git):  # worktree / submodule: "gitdir: <path>"
        git = os.path.join(repo, open(git).read().split(":", 1)[1].strip())
    head = open(os.path.join(git, "HEAD")).read().strip()
    if not head.startswith("ref:"):
        return head  # detached HEAD
    ref = head.split(None, 1)[1]
    loose = os.path.join(git, ref)
    if os.path.isfile(loose):
        return open(loose).read().strip()
    packed = os.path.join(git, "packed-refs")
    if os.path.isfile(packed):
        for line in open(packed):
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    raise RuntimeError("cannot resolve %s in %s" % (ref, git))


def branch(repo="."):
    try:
        head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        return head.split("refs/heads/", 1)[1] if "refs/heads/" in head else None
    except OSError:
        return None


if __name__ == "__main__":
    try:
        print(commit(sys.argv[1] if len(sys.argv) > 1 else "."))
    except Exception as e:
        print("unknown (%s)" % e, file=sys.stderr)
        sys.exit(1)
