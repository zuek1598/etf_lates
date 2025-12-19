#!/usr/bin/env python3
"""
Safe pull script for ASX-M-M project
Safely pulls changes from GitHub without losing local work
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

def safe_pull():
    """Safely pull changes from GitHub"""
    
    print("🔄 Checking for local changes...")
    
    # Check if there are uncommitted changes
    success, stdout, stderr = run_git_command("git status --porcelain")
    if not success:
        print(f"❌ Error checking git status: {stderr}")
        return False
    
    if stdout.strip():
        print("⚠️  You have uncommitted changes:")
        print(stdout)
        
        choice = input("\nChoose an option:\n1. Commit changes first (recommended)\n2. Stash changes temporarily\n3. Cancel pull\n\nEnter choice (1-3): ")
        
        if choice == "1":
            print("📝 Committing local changes...")
            success, _, _ = run_git_command("git add .")
            if not success:
                print("❌ Error adding files")
                return False
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            success, _, _ = run_git_command(f'git commit -m "Local changes before pull: {timestamp}"')
            if not success:
                print("❌ Error committing changes")
                return False
            
            print("✅ Local changes committed")
            
        elif choice == "2":
            print("💾 Stashing local changes...")
            success, _, _ = run_git_command("git stash")
            if not success:
                print("❌ Error stashing changes")
                return False
            print("✅ Changes stashed temporarily")
            
        else:
            print("❌ Pull cancelled")
            return False
    
    # Pull changes from GitHub
    print("\n🚀 Pulling changes from GitHub...")
    success, stdout, stderr = run_git_command("git pull origin main")
    
    if not success:
        print(f"❌ Error pulling changes: {stderr}")
        
        # Check if it's a merge conflict
        if "conflict" in stderr.lower():
            print("⚠️  Merge conflict detected. Please resolve conflicts manually.")
            print("💡 Use 'git status' to see conflicting files")
        
        return False
    
    print("✅ Successfully pulled changes from GitHub!")
    
    # Restore stashed changes if they were stashed
    if stdout.strip() and "stash" in locals():
        print("\n💾 Restoring stashed changes...")
        success, _, stderr = run_git_command("git stash pop")
        if success:
            print("✅ Stashed changes restored")
        else:
            print(f"⚠️  Could not restore stashed changes: {stderr}")
            print("💡 Run 'git stash pop' manually after resolving any conflicts")
    
    return True

if __name__ == "__main__":
    safe_pull()
