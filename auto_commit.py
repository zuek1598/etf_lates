#!/usr/bin/env python3
"""
Auto-commit script for ASX-M-M project
Run this to automatically commit and push changes
"""

import subprocess
import os
from datetime import datetime

def run_git_command(command):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=".")
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def auto_commit_and_push():
    """Automatically commit and push changes"""
    
    # Check if there are changes
    success, stdout, stderr = run_git_command("git status --porcelain")
    if not success:
        print(f"❌ Error checking git status: {stderr}")
        return False
    
    if not stdout.strip():
        print("✅ No changes to commit")
        return True
    
    print("🔄 Changes detected, proceeding with auto-commit...")
    
    # Add all changes
    success, stdout, stderr = run_git_command("git add .")
    if not success:
        print(f"❌ Error adding files: {stderr}")
        return False
    
    # Commit with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_message = f"Auto-update: {timestamp}"
    
    success, stdout, stderr = run_git_command(f'git commit -m "{commit_message}"')
    if not success:
        print(f"❌ Error committing: {stderr}")
        return False
    
    print(f"✅ Committed changes: {commit_message}")
    
    # Push to GitHub
    success, stdout, stderr = run_git_command("git push origin main")
    if not success:
        print(f"❌ Error pushing: {stderr}")
        return False
    
    print("🚀 Successfully pushed to GitHub!")
    return True

if __name__ == "__main__":
    auto_commit_and_push()
