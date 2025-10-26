#!/usr/bin/env python3
"""Simple import debug script to test individual module imports."""

import sys
import traceback


def test_import(module_name: str) -> None:
    """Test importing a specific module with detailed error reporting."""
    print(f"\n{'='*50}")
    print(f"Testing import: {module_name}")
    print('='*50)
    
    try:
        # Try to import the module
        module = __import__(module_name)
        print(f"✅ SUCCESS: {module_name} imported successfully")
        
        # Print module information
        if hasattr(module, '__file__'):
            print(f"   📁 Location: {module.__file__}")
        if hasattr(module, '__version__'):
            print(f"   🏷️  Version: {module.__version__}")
        if hasattr(module, '__path__'):
            print(f"   📂 Package path: {module.__path__}")
            
        # For runpod, try to access common attributes
        if module_name == 'runpod':
            print("   🔍 Checking RunPod attributes:")
            attrs = ['api_key', 'endpoint', 'Endpoint', 'serverless', 'get_gpu']
            for attr in attrs:
                if hasattr(module, attr):
                    print(f"      ✅ {attr}")
                else:
                    print(f"      ❌ {attr} (missing)")
                    
    except ImportError as e:
        print(f"❌ IMPORT ERROR: {e}")
        print(f"   💡 This usually means the package is not installed")
        
        # Try to provide more specific information
        if "No module named" in str(e):
            missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
            print(f"   🔍 Missing module: {missing_module}")
            
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        print(f"   📋 Full traceback:")
        traceback.print_exc()


def main():
    """Test importing critical modules for SkyPilot RunPod."""
    print("🔍 Import Debug Script")
    print("Testing critical imports for SkyPilot RunPod integration")
    
    # List of modules to test
    modules_to_test = [
        'runpod',
        'sky',
        'skypilot',
        'sky.clouds',
        'sky.clouds.runpod',
        'sky.provision',
        'sky.provision.runpod'
    ]
    
    for module in modules_to_test:
        test_import(module)
    
    print(f"\n{'='*50}")
    print("🏁 Import testing complete!")
    print('='*50)


if __name__ == "__main__":
    main()
