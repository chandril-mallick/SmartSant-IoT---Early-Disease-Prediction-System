#!/bin/bash

# SmartSant-IoT Auto Git Push Script
# This script automatically stages, commits, and pushes changes to GitHub

echo "🔍 Checking git status..."
git status

echo ""
echo "📝 Staging all changes..."
git add .

echo ""
read -p "Enter commit message (or press Enter for default): " commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Update: project improvements on $(date '+%Y-%m-%d %H:%M')"
fi

echo ""
echo "💾 Committing changes with message: '$commit_msg'"
git commit -m "$commit_msg"

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Pushing to GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully pushed to GitHub!"
        echo "🔗 View at: https://github.com/chandril-mallick/SmartSant-IoT---Early-Disease-Prediction-System"
    else
        echo ""
        echo "❌ Push failed. Please check the error message above."
    fi
else
    echo ""
    echo "⚠️  Nothing to commit or commit failed."
fi
