#!/usr/bin/env bash

set -euo pipefail

message="${1:-}"
if [[ -z "${message// }" ]]; then
  echo 'Usage: make ship MESSAGE="Describe the completed change"' >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "This command must run inside a Git repository." >&2
  exit 2
}
cd "$repo_root"

author_name="$(git config user.name || true)"
author_email="$(git config user.email || true)"
if [[ -z "$author_name" || -z "$author_email" ]]; then
  echo "Git user.name and user.email must be configured before shipping." >&2
  exit 2
fi

remote_url="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$remote_url" ]]; then
  echo "The repository does not have an origin remote." >&2
  exit 2
fi

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  echo "Cannot ship from a detached HEAD." >&2
  exit 2
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Nothing to commit. Make a meaningful change before shipping."
  exit 0
fi

candidate_files="$({
  git diff --name-only
  git diff --cached --name-only
  git ls-files --others --exclude-standard
} | sort -u)"

blocked_files=()
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  case "$file" in
    .env.example|*/.env.example)
      continue
      ;;
    .env|.env.*|*/.env|*/.env.*|*.pem|*.key|*.p12|*.pfx|*credentials*|*secret*)
      blocked_files+=("$file")
      ;;
  esac
done <<< "$candidate_files"

if (( ${#blocked_files[@]} > 0 )); then
  echo "Refusing to stage potentially sensitive files:" >&2
  printf '  %s\n' "${blocked_files[@]}" >&2
  exit 3
fi

echo "Shipping as $author_name <$author_email> on branch $branch"
echo "Running repository checks and refreshing tracker..."
make ci
.venv/bin/python -m scripts.update_tracker --checks-passed
git diff --check

git add -A
if git diff --cached --quiet; then
  echo "No committable changes remain after checks."
  exit 0
fi

git commit -m "$message"
git push -u origin "$branch"

default_ref="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null || true)"
default_branch="${default_ref#origin/}"
if [[ -n "$default_branch" && "$branch" != "$default_branch" ]]; then
  echo
  echo "Pushed $branch successfully. GitHub counts its commits after they reach $default_branch."
  echo "Open or update a pull request, review it, and merge when the work is ready."
else
  echo "Pushed $branch successfully."
fi
