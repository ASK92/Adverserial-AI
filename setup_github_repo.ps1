# PowerShell script to set up and push to GitHub repository
# Repository name: "Adverserial-AI"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Setup Script" -ForegroundColor Cyan
Write-Host "Repository: Adverserial-AI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "[OK] Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Check if already a git repository
if (Test-Path .git) {
    Write-Host "[INFO] Git repository already initialized" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "[OK] Git repository initialized" -ForegroundColor Green
}

# Add all files
Write-Host "[INFO] Adding files to git..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host "[INFO] Committing files..." -ForegroundColor Yellow
    
    # Set git user if not already set (optional)
    $gitUser = git config user.name
    $gitEmail = git config user.email
    
    if (-not $gitUser) {
        Write-Host "[INFO] Git user name not set. Please set it:" -ForegroundColor Yellow
        $userName = Read-Host "Enter your name"
        $userEmail = Read-Host "Enter your email"
        git config user.name $userName
        git config user.email $userEmail
    }
    
    git commit -m "Initial commit: Adversarial Patch Pipeline with Cyberphysical Attack System"
    Write-Host "[OK] Files committed" -ForegroundColor Green
} else {
    Write-Host "[INFO] No changes to commit" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor White
Write-Host "   - Go to: https://github.com/new" -ForegroundColor Gray
Write-Host "   - Repository name: Adverserial-AI" -ForegroundColor Gray
Write-Host "   - Description: Adversarial Patch Pipeline for Computer Vision Defense Bypass" -ForegroundColor Gray
Write-Host "   - Choose Public or Private" -ForegroundColor Gray
Write-Host "   - DO NOT initialize with README, .gitignore, or license" -ForegroundColor Yellow
Write-Host "   - Click 'Create repository'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. After creating the repository, run these commands:" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/Adverserial-AI.git" -ForegroundColor Green
Write-Host "   git branch -M main" -ForegroundColor Green
Write-Host "   git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "   (Replace YOUR_USERNAME with your GitHub username)" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or use SSH (if you have SSH keys set up):" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin git@github.com:YOUR_USERNAME/Adverserial-AI.git" -ForegroundColor Green
Write-Host "   git branch -M main" -ForegroundColor Green
Write-Host "   git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to set up remote now
$setupRemote = Read-Host "Do you want to set up the remote repository now? (y/n)"
if ($setupRemote -eq "y" -or $setupRemote -eq "Y") {
    $username = Read-Host "Enter your GitHub username"
    $useSSH = Read-Host "Use SSH? (y/n) [default: n]"
    
    if ($useSSH -eq "y" -or $useSSH -eq "Y") {
        $remoteUrl = "git@github.com:$username/Adverserial-AI.git"
    } else {
        $remoteUrl = "https://github.com/$username/Adverserial-AI.git"
    }
    
    Write-Host "[INFO] Adding remote origin..." -ForegroundColor Yellow
    git remote add origin $remoteUrl 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Remote added: $remoteUrl" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Remote might already exist. Checking..." -ForegroundColor Yellow
        git remote set-url origin $remoteUrl
        Write-Host "[OK] Remote updated: $remoteUrl" -ForegroundColor Green
    }
    
    Write-Host "[INFO] Setting branch to main..." -ForegroundColor Yellow
    git branch -M main
    
    Write-Host ""
    Write-Host "Ready to push! Run this command:" -ForegroundColor Cyan
    Write-Host "  git push -u origin main" -ForegroundColor Green
    Write-Host ""
    
    $pushNow = Read-Host "Push to GitHub now? (y/n)"
    if ($pushNow -eq "y" -or $pushNow -eq "Y") {
        Write-Host "[INFO] Pushing to GitHub..." -ForegroundColor Yellow
        git push -u origin main
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Successfully pushed to GitHub!" -ForegroundColor Green
            Write-Host "Repository URL: https://github.com/$username/Adverserial-AI" -ForegroundColor Cyan
        } else {
            Write-Host "[ERROR] Push failed. Please check your credentials and try again." -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
