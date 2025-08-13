#!/usr/bin/env python3
"""
Script to update embeddings and related files when new images are added to the output folder.
This script will:
1. Sync image_paths.json with actual images in output folder
2. Rebuild FAISS indices if needed
3. Generate captions for new images
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_sync_needed():
    """Check if image_paths.json needs to be synced."""
    try:
        # Import the sync functions
        sys.path.append('src')
        from sync_image_paths import get_all_images_in_output, load_current_image_paths
        
        actual_images = get_all_images_in_output()
        current_images = load_current_image_paths()
        
        return len(actual_images) != len(current_images) or set(actual_images) != set(current_images)
    except Exception as e:
        print(f"⚠️ Could not check sync status: {e}")
        return True  # Assume sync is needed if we can't check

def main():
    print("🔧 Embeddings Update Tool")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('output') or not os.path.exists('src'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Sync image paths
    if check_sync_needed():
        print("📝 Image paths need to be synchronized...")
        if not run_command("python src/sync_image_paths.py", "Syncing image paths"):
            print("❌ Failed to sync image paths. Please run manually: python src/sync_image_paths.py")
            return False
    else:
        print("✅ Image paths are already in sync")
    
    # Step 2: Ask user what they want to update
    print("\n🔄 What would you like to update?")
    print("1. Generate captions for new images")
    print("2. Rebuild FAISS indices")
    print("3. Both captions and indices")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1" or choice == "3":
        print("\n📝 Generating captions...")
        if not run_command("python src/generate_captions_simple.py", "Generating captions"):
            print("⚠️ Caption generation failed, but continuing...")
    
    if choice == "2" or choice == "3":
        print("\n🔍 Rebuilding FAISS indices...")
        
        # Check which embedding scripts are available
        embedding_scripts = []
        if os.path.exists("src/embedding_fashionclip.py"):
            embedding_scripts.append(("src/embedding_fashionclip.py", "FashionCLIP embeddings"))
        if os.path.exists("src/embedding_openclip.py"):
            embedding_scripts.append(("src/embedding_openclip.py", "OpenCLIP embeddings"))
        
        for script, description in embedding_scripts:
            run_command(f"python {script}", f"Building {description}")
        
        # Build FAISS index
        if os.path.exists("src/build_faiss_index.py"):
            run_command("python src/build_faiss_index.py", "Building FAISS index")
    
    if choice == "4":
        print("👋 Exiting...")
        return True
    
    print("\n✨ Update process completed!")
    print("\n📊 Current status:")
    
    # Show final statistics
    try:
        sys.path.append('src')
        from sync_image_paths import get_all_images_in_output, show_directory_breakdown
        
        total_images = len(get_all_images_in_output())
        print(f"   Total images: {total_images}")
        
        # Check if caption file exists
        if os.path.exists("data/captions/simple_captions.json"):
            with open("data/captions/simple_captions.json", 'r') as f:
                captions = json.load(f)
                print(f"   Images with captions: {len(captions)}")
        
        # Check if embeddings exist
        embedding_files = [
            "data/embeddings/fashionclip_index.faiss",
            "data/embeddings/openclip_index.faiss"
        ]
        
        for emb_file in embedding_files:
            if os.path.exists(emb_file):
                print(f"   ✅ {os.path.basename(emb_file)} exists")
            else:
                print(f"   ❌ {os.path.basename(emb_file)} missing")
                
    except Exception as e:
        print(f"   ⚠️ Could not get status: {e}")
    
    return True

if __name__ == "__main__":
    main()
