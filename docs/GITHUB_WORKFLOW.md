# GitHub contribution workflow

Use the repository's shipping command to turn completed work into a tested commit without
typing several Git commands:

```bash
make ship MESSAGE="Improve voice signal animation"
```

The command:

1. verifies the configured Git author and `origin` remote;
2. blocks filenames that look like secrets or credentials;
3. runs `make ci`;
4. stages the repository changes;
5. creates one commit using the supplied message; and
6. pushes the current branch to `origin`.

## Contribution graph requirements

GitHub associates a commit with your profile when its author email belongs to your GitHub
account. Commits must ultimately reach the repository's default branch (or `gh-pages`) to appear
on the contribution graph. Timbre's current default branch is `prod`, so commits shipped from a
feature branch such as `ui` appear after the branch is merged.

Prefer one focused commit for a completed change. Do not generate empty commits, backdate
commits, or commit every file save solely to inflate the graph. Recruiters can inspect the
history, tests, pull requests, and quality of the work behind the activity.

## Before the first push

Confirm that this email is added and verified on your GitHub account:

```bash
git config user.email
```

This repository currently uses a GitHub `noreply` address, which is suitable when it belongs to
the same GitHub account that owns the repository.
