---
description: Push project updates to GitHub
---

# Git Push Workflow

This workflow helps you automatically push your project updates to GitHub.

## Steps

### 1. Check current status
First, check what files have been modified:
```bash
git status
```

### 2. Add all changes
Add all modified, new, and deleted files to staging:
// turbo
```bash
git add .
```

### 3. Commit changes
Commit the changes with a descriptive message. Replace the message with your own:
```bash
git commit -m "Update: [describe your changes here]"
```

**Common commit message examples:**
- `git commit -m "Add: new kidney disease classifier"`
- `git commit -m "Fix: urine model preprocessing bug"`
- `git commit -m "Update: improved stool classification accuracy"`
- `git commit -m "Docs: updated README with installation instructions"`

### 4. Push to GitHub
Push the committed changes to the remote repository:
// turbo
```bash
git push origin main
```

## Quick One-Liner (for simple updates)

If you want to quickly commit and push all changes with a generic message:
```bash
git add . && git commit -m "Update: project improvements" && git push origin main
```

## Tips

- **Before pushing**: Always review your changes with `git status` and `git diff`
- **Commit messages**: Write clear, descriptive commit messages
- **Large files**: Remember that files over 100MB cannot be pushed to GitHub
- **Sensitive data**: Never commit API keys, passwords, or sensitive information
- **Virtual environments**: The `.gitignore` file already excludes `venv/` and model files

## Troubleshooting

**If push is rejected:**
```bash
git pull origin main --rebase
git push origin main
```

**If you need to undo the last commit (before pushing):**
```bash
git reset --soft HEAD~1
```

**If you accidentally committed large files:**
```bash
git rm --cached path/to/large/file
git commit --amend -m "Remove large file"
git push origin main --force
```
