import os
import json
from pathlib import Path

def get_all_images_in_output(output_dir="output"):
    """Get all image files in the output directory and subdirectories."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
    image_paths = []
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory '{output_dir}' does not exist")
        return []
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions:
                # Convert to forward slashes for consistency
                relative_path = str(file_path).replace('\\', '/')
                image_paths.append(relative_path)
    
    return sorted(image_paths)

def load_current_image_paths(json_path="data/embeddings/image_paths.json"):
    """Load the current image paths from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File '{json_path}' not found")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON file: {e}")
        return []

def save_image_paths(image_paths, json_path="data/embeddings/image_paths.json"):
    """Save image paths to JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(image_paths, f, indent=2, ensure_ascii=False)

def compare_and_sync_image_paths():
    """Compare current image_paths.json with actual files and sync them."""
    print("🔍 Scanning output directory for images...")
    actual_images = get_all_images_in_output()
    
    print("📂 Loading current image_paths.json...")
    current_images = load_current_image_paths()
    
    print(f"\n📊 Comparison Results:")
    print(f"   Images in output folder: {len(actual_images)}")
    print(f"   Images in image_paths.json: {len(current_images)}")
    
    # Find differences
    actual_set = set(actual_images)
    current_set = set(current_images)
    
    missing_from_json = actual_set - current_set
    extra_in_json = current_set - actual_set
    
    if missing_from_json:
        print(f"\n⚠️  Images in output folder but NOT in image_paths.json ({len(missing_from_json)}):")
        for img in sorted(missing_from_json)[:10]:  # Show first 10
            print(f"   + {img}")
        if len(missing_from_json) > 10:
            print(f"   ... and {len(missing_from_json) - 10} more")
    
    if extra_in_json:
        print(f"\n⚠️  Images in image_paths.json but NOT in output folder ({len(extra_in_json)}):")
        for img in sorted(extra_in_json)[:10]:  # Show first 10
            print(f"   - {img}")
        if len(extra_in_json) > 10:
            print(f"   ... and {len(extra_in_json) - 10} more")
    
    if not missing_from_json and not extra_in_json:
        print("\n✅ image_paths.json is already in sync with output folder!")
        return True
    
    # Ask user if they want to sync
    print(f"\n🔄 Would you like to update image_paths.json to match the output folder?")
    print(f"   This will:")
    if missing_from_json:
        print(f"   • Add {len(missing_from_json)} missing images")
    if extra_in_json:
        print(f"   • Remove {len(extra_in_json)} non-existent images")
    
    response = input("\nProceed with sync? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n🔄 Updating image_paths.json...")
        save_image_paths(actual_images)
        print("✅ image_paths.json has been updated!")
        
        # Verify the update
        updated_images = load_current_image_paths()
        if set(updated_images) == actual_set:
            print("✅ Verification successful - files are now in sync!")
        else:
            print("❌ Verification failed - there may be an issue with the update")
        
        return True
    else:
        print("❌ Sync cancelled by user")
        return False

def show_directory_breakdown():
    """Show breakdown of images by directory."""
    actual_images = get_all_images_in_output()
    
    # Group by directory
    dir_counts = {}
    for img_path in actual_images:
        directory = os.path.dirname(img_path)
        dir_counts[directory] = dir_counts.get(directory, 0) + 1
    
    print(f"\n📁 Images by directory:")
    for directory, count in sorted(dir_counts.items()):
        print(f"   {directory}: {count} images")
    
    return dir_counts

if __name__ == "__main__":
    print("🔧 Image Paths Synchronization Tool")
    print("=" * 50)
    
    # Show directory breakdown
    show_directory_breakdown()
    
    # Compare and sync
    compare_and_sync_image_paths()
    
    print("\n✨ Done!")
