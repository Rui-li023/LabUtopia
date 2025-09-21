#!/usr/bin/env python3
"""
Auto-setup script for VS Code settings.json with Isaac Sim paths
"""

import os
import json
import sys
from pathlib import Path

def find_python_executable():
    """Find the current Python executable path"""
    return sys.executable

def find_conda_env_path():
    """Find the conda environment path"""
    python_path = find_python_executable()
    print(f"Python path: {python_path}")
    # Extract the environment path from python executable
    # e.g., /home/ubuntu/mambaforge/envs/isaaclab/bin/python -> /home/ubuntu/mambaforge/envs/isaaclab
    env_path = Path(python_path).parent.parent
    return str(env_path)

def find_isaac_sim_path():
    """Find Isaac Sim installation path"""
    # Common Isaac Sim installation paths
    possible_paths = [
        "~/.local/share/ov/pkg/isaac-sim-4.2.0",
        "~/.local/share/ov/pkg/isaac-sim-4.1.0",
        "~/.local/share/ov/pkg/isaac-sim-4.0.0",
        "~/.local/share/ov/pkg/isaac-sim-2023.1.1",
        "~/.local/share/ov/pkg/isaac-sim-2023.1.0",
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    # If not found, ask user
    print("Isaac Sim installation not found in common locations.")
    user_path = input("Please enter the Isaac Sim installation path: ")
    if os.path.exists(user_path):
        return user_path
    else:
        print(f"Path {user_path} does not exist!")
        return None

def generate_settings_json():
    """Generate the VS Code settings.json content"""
    
    # Get paths
    conda_env_path = find_conda_env_path()
    print(f"Python environment: {conda_env_path}")
    
    # Build the site-packages path
    site_packages = os.path.join(conda_env_path, "lib/python3.10/site-packages")
    print(f"Site packages: {site_packages}")
    # Generate paths list
    extra_paths = [
        f"{site_packages}/isaacsim",
        f"{site_packages}/omni",
        f"{site_packages}/isaacsim/exts/omni.isaac.kit",
        f"{site_packages}/isaacsim/exts/omni.isaac.core",
        f"{site_packages}/isaacsim/exts/omni.isaac.sensor",
        f"{site_packages}/isaacsim/exts/omni.isaac.franka",
        f"{site_packages}/isaacsim/exts/omni.isaac.nucleus",
        f"{site_packages}/isaacsim/exts/omni.isaac.motion_generation",
        f"{site_packages}/isaacsim/exts/omni.usd.libs",
        f"{site_packages}/isaacsim/exts/omni.isaac.manipulators",
        f"{site_packages}/isaacsim/extsPhysics/omni.physx",
        f"{site_packages}/isaacsim/extsPhysics/omni.physx.tensors",
        f"{site_packages}/isaacsim/extscache/omni.usd-1.12.2+10a4b5c0.lx64.r.cp310",
        f"{site_packages}/isaacsim/exts/omni.replicator.isaac",
        f"{site_packages}/isaacsim/omni.isaac.kit"
    ]
    
    # Filter out non-existent paths
    valid_paths = [path for path in extra_paths if os.path.exists(path)]
    
    if len(valid_paths) != len(extra_paths):
        print(f"Warning: {len(extra_paths) - len(valid_paths)} paths do not exist")
    
    # Create settings dictionary
    settings = {
        "python.analysis.extraPaths": valid_paths,
        "python.analysis.typeCheckingMode": "off",
        "files.associations": {
            "concepts": "cpp",
            "exception": "cpp",
            "type_traits": "cpp",
            "new": "cpp",
            "typeinfo": "cpp"
        }
    }
    
    return settings

def create_vscode_settings(settings_dict):
    """Create the .vscode/settings.json file"""
    
    # Create .vscode directory if it doesn't exist
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Write settings.json
    settings_file = vscode_dir / "settings.json"
    
    with open(settings_file, 'w') as f:
        json.dump(settings_dict, f, indent=4)
    
    print(f"Settings written to {settings_file}")

def main():
    """Main function"""
    print("=== VS Code Settings.json Auto-Setup for Isaac Sim ===\n")
    
    # Generate settings
    settings = generate_settings_json()
    
    if settings is None:
        print("Failed to generate settings")
        return 1
    
    # Create the file
    create_vscode_settings(settings)
    
    print("\nSetup complete! VS Code should now have proper Isaac Sim paths configured.")
    print("Restart VS Code to apply the changes.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())