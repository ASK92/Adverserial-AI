#!/bin/bash
# Bash script to set up and push to GitHub repository
# Repository name: "Adverserial-AI"

echo "========================================"
echo "GitHub Repository Setup Script"
echo "Repository: Adverserial-AI"
echo "========================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git is not installed. Please install Git first."
    exit 1
fi

echo "[OK] Git is installed: $(git --version)"

# Check if already a git repository
if [ -d .git ]; then
    echo "[INFO] Git repository already initialized"
else
    echo "[INFO] Initializing git repository..."
    git init
    echo "[OK] Git repository initialized"
fi

# Add all files
echo "[INFO] Adding files to git..."
git add .

# Check if there are changes to commit
if [ -n "$(git status --porcelain)" ]; then
    echo "[INFO] Committing files..."
    
    # Check if git user is set
    if [ -z "$(git config user.name)" ]; then
        echo "[INFO] Git user name not set. Please set it:"
        read -p "Enter your name: " userName
        read -p "Enter your email: " userEmail
        git config user.name "$userName"
        git config user.email "$userEmail"
    fi
    
    git commit -m "Initial commit: Adversarial Patch Pipeline with Cyberphysical Attack System"
    echo "[OK] Files committed"
else
    echo "[INFO] No changes to commit"
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: Adverserial-AI"
echo "   - Description: Adversarial Patch Pipeline for Computer Vision Defense Bypass"
echo "   - Choose Public or Private"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo "   - Click 'Create repository'"
echo ""
echo "2. After creating the repository, run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/Adverserial-AI.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "   (Replace YOUR_USERNAME with your GitHub username)"
echo ""
echo "3. Or use SSH (if you have SSH keys set up):"
echo ""
echo "   git remote add origin git@github.com:YOUR_USERNAME/Adverserial-AI.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "========================================"
echo ""

# Ask if user wants to set up remote now
read -p "Do you want to set up the remote repository now? (y/n): " setupRemote
if [ "$setupRemote" = "y" ] || [ "$setupRemote" = "Y" ]; then
    read -p "Enter your GitHub username: " username
    read -p "Use SSH? (y/n) [default: n]: " useSSH
    
    if [ "$useSSH" = "y" ] || [ "$useSSH" = "Y" ]; then
        remoteUrl="git@github.com:$username/Adverserial-AI.git"
    else
        remoteUrl="https://github.com/$username/Adverserial-AI.git"
    fi
    
    echo "[INFO] Adding remote origin..."
    git remote add origin "$remoteUrl" 2>/dev/null || git remote set-url origin "$remoteUrl"
    echo "[OK] Remote added: $remoteUrl"
    
    echo "[INFO] Setting branch to main..."
    git branch -M main
    
    echo ""
    echo "Ready to push! Run this command:"
    echo "  git push -u origin main"
    echo ""
    
    read -p "Push to GitHub now? (y/n): " pushNow
    if [ "$pushNow" = "y" ] || [ "$pushNow" = "Y" ]; then
        echo "[INFO] Pushing to GitHub..."
        git push -u origin main
        if [ $? -eq 0 ]; then
            echo "[OK] Successfully pushed to GitHub!"
            echo "Repository URL: https://github.com/$username/Adverserial-AI"
        else
            echo "[ERROR] Push failed. Please check your credentials and try again."
        fi
    fi
fi

echo ""
echo "Setup complete!"
