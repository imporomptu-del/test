#!/usr/bin/env python3
"""
KOLOMVERSE Dataset Pipeline
Extracts images from ZIP files and converts labels to YOLO format for training YOLO11n.
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
from collections import defaultdict
import tempfile

class KolomverseDatasetPipeline:
    def __init__(self):
        # Source paths
        self.source_zip_dir = Path("/Volumes/T7/KOLOMVERSE/KOLOMVERSE/img_zip/zip")
        self.data_dir = Path("/Users/romanmaksymiuk/Documents/SEAQR/data")
        
        # Output paths
        self.output_dir = Path("/Volumes/T7/KOLOMVERSE/KOLOMVERSE/img_zip/zip/ready_to_upload/dataset")
        
        # Class mapping from your data.yaml
        self.class_map = {
            "buoy": 0,
            "fishnet buoy": 1, 
            "lighthouse": 2,
            "ship": 3,
            "wind farm": 4
        }
        
        self.splits = ["train", "test", "validation"]
        
    def setup_directories(self):
        """Create the YOLO dataset directory structure"""
        print("🏗️  Setting up directory structure...")
        
        # Create main directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directories
        for split in self.splits:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Created directory structure at: {self.output_dir}")
        
    def load_unique_images(self, split):
        """Load unique image paths for a given split"""
        unique_file = self.data_dir / f"{split}_unique.txt"
        
        if not unique_file.exists():
            raise FileNotFoundError(f"Unique file not found: {unique_file}")
            
        with open(unique_file, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
            
        print(f"📋 Loaded {len(paths)} unique images for {split}")
        return paths
        
    def extract_images_from_zips(self, split):
        """Extract images for a given split from ZIP files"""
        print(f"📦 Extracting {split} images from ZIP files...")
        
        unique_paths = self.load_unique_images(split)
        split_zip_dir = self.source_zip_dir / split
        output_images_dir = self.output_dir / "images" / split
        
        # Group images by ZIP file number
        zip_groups = defaultdict(list)
        for path in unique_paths:
            # Extract zip number from path like "train/0/0000000005.jpg"
            parts = path.split('/')
            if len(parts) >= 3:
                zip_num = parts[1]  # Get the zip number (0, 1, 2, etc.)
                zip_groups[zip_num].append(path)
                
        total_extracted = 0
        
        for zip_num, image_paths in tqdm(zip_groups.items(), desc=f"Processing {split} ZIPs"):
            zip_file_path = split_zip_dir / f"{zip_num}.zip"
            
            if not zip_file_path.exists():
                print(f"⚠️  Warning: ZIP file not found: {zip_file_path}")
                continue
                
            # Extract images from this ZIP using the same method as your bash script
            extracted_count = self._extract_from_single_zip_bash_style(
                zip_file_path, image_paths, output_images_dir
            )
            total_extracted += extracted_count
            
        print(f"✅ Extracted {total_extracted} images for {split}")
        return total_extracted
        
    def _extract_from_single_zip_bash_style(self, zip_path, image_paths, output_dir):
        """Extract specific images from a single ZIP file using bash-style approach"""
        extracted_count = 0
        
        try:
            # Create a temporary file with images that exist in this zip
            temp_list = tempfile.NamedTemporaryFile(mode='w', delete=False)
            
            # Check which images from our list exist in this zip (like your bash script)
            for image_path in image_paths:
                try:
                    # Use unzip -l to list contents and grep to check if image exists
                    cmd = f'unzip -l "{zip_path}" | grep -q "{image_path}"'
                    result = subprocess.run(cmd, shell=True, capture_output=True)
                    
                    if result.returncode == 0:  # grep found the image
                        temp_list.write(f"{image_path}\n")
                        
                except Exception as e:
                    print(f"⚠️  Error checking {image_path}: {e}")
                    continue
            
            temp_list.close()
            
            # Extract all matching images at once (like your bash script)
            if os.path.getsize(temp_list.name) > 0:
                # Read the temp file to get list of images to extract
                with open(temp_list.name, 'r') as f:
                    images_to_extract = [line.strip() for line in f if line.strip()]
                
                # Extract each image using unzip -j (flatten directory structure)
                for image_path in images_to_extract:
                    try:
                        # Use unzip -j to extract and flatten path, like: unzip -j zipfile "*image_path" -d output_dir
                        cmd = [
                            'unzip', '-j', str(zip_path), 
                            f"*{image_path}", 
                            '-d', str(output_dir)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            extracted_count += 1
                        else:
                            print(f"⚠️  Failed to extract {image_path}: {result.stderr}")
                            
                    except Exception as e:
                        print(f"⚠️  Error extracting {image_path}: {e}")
                        
                print(f"  📸 Extracted {extracted_count} images from {zip_path.name}")
            else:
                print(f"  ❌ No matching images found in {zip_path.name}")
            
            # Clean up temp file
            os.unlink(temp_list.name)
            
        except Exception as e:
            print(f"❌ Error processing {zip_path}: {e}")
            # Clean up temp file even on error
            try:
                os.unlink(temp_list.name)
            except:
                pass
            
        return extracted_count
        
    def convert_labels_to_yolo(self, split):
        """Convert CSV labels to YOLO format for a given split"""
        print(f"🏷️  Converting {split} labels to YOLO format...")
        
        csv_path = self.data_dir / f"{split}_labels.csv"
        labels_output_dir = self.output_dir / "labels" / split
        
        if not csv_path.exists():
            print(f"⚠️  Warning: Labels file not found: {csv_path}")
            return 0
            
        # Load CSV
        df = pd.read_csv(csv_path)
        df = df[df["label"].notna()]  # Remove rows with no label
        
        converted_count = 0
        
        # Process each image's labels
        for image_path, group in tqdm(df.groupby("image"), desc=f"Converting {split} labels"):
            # Get image dimensions
            width = group.iloc[0]["width"]
            height = group.iloc[0]["height"]
            
            # Convert each annotation for this image
            yolo_lines = []
            for _, row in group.iterrows():
                if row["label"] not in self.class_map:
                    print(f"⚠️  Unknown class: {row['label']}")
                    continue
                    
                class_id = self.class_map[row["label"]]
                
                # Convert bounding box to YOLO format
                xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                
                # Calculate center coordinates and dimensions (normalized)
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
            
            # Save YOLO format file
            if yolo_lines:
                image_filename = os.path.basename(image_path)
                label_filename = os.path.splitext(image_filename)[0] + ".txt"
                label_path = labels_output_dir / label_filename
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                    
                converted_count += 1
                
        print(f"✅ Converted {converted_count} label files for {split}")
        return converted_count
        
    def create_data_yaml(self):
        """Create the data.yaml file for YOLO training"""
        print("📝 Creating data.yaml file...")
        
        # Create data.yaml content
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/validation', 
            'test': 'images/test',
            'nc': len(self.class_map),
            'names': list(self.class_map.keys())
        }
        
        # Save data.yaml
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
            
        print(f"✅ Created data.yaml at: {yaml_path}")
        
    def verify_dataset(self):
        """Verify that images and labels match correctly"""
        print("🔍 Verifying dataset integrity...")
        
        for split in self.splits:
            images_dir = self.output_dir / "images" / split
            labels_dir = self.output_dir / "labels" / split
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"⚠️  {split} directories not found")
                continue
                
            image_files = set(f.stem for f in images_dir.glob("*.jpg"))
            label_files = set(f.stem for f in labels_dir.glob("*.txt"))
            
            missing_labels = image_files - label_files
            missing_images = label_files - image_files
            
            print(f"📊 {split.upper()} Split:")
            print(f"   Images: {len(image_files)}")
            print(f"   Labels: {len(label_files)}")
            print(f"   Missing labels: {len(missing_labels)}")
            print(f"   Missing images: {len(missing_images)}")
            
            if missing_labels:
                print(f"   ⚠️  Some images missing labels: {list(missing_labels)[:5]}...")
            if missing_images:
                print(f"   ⚠️  Some labels missing images: {list(missing_images)[:5]}...")
                
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("🚀 Starting KOLOMVERSE Dataset Pipeline...")
        print("="*60)
        
        try:
            # Step 1: Setup directories
            self.setup_directories()
            
            # Step 2: Extract images for each split
            for split in self.splits:
                self.extract_images_from_zips(split)
                
            # Step 3: Convert labels for each split  
            for split in self.splits:
                self.convert_labels_to_yolo(split)
                
            # Step 4: Create data.yaml
            self.create_data_yaml()
            
            # Step 5: Verify dataset
            self.verify_dataset()
            
            print("="*60)
            print("🎉 Pipeline completed successfully!")
            print(f"📁 Dataset ready at: {self.output_dir}")
            print("🚀 Ready to upload to GCS and train YOLO11n!")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    pipeline = KolomverseDatasetPipeline()
    pipeline.run_pipeline()
