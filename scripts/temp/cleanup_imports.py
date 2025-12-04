import os
import glob
import sys

def cleanup_sys_path_hacks():
    """
    This script removes all 'sys.path.append' lines from all .py files
    in the project directory, except for the 'venv' folder.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Starting cleanup in directory: {project_dir}")
    
    # Find all python files recursively
    python_files = glob.glob(os.path.join(project_dir, '**', '*.py'), recursive=True)
    
    modified_files_count = 0

    for file_path in python_files:
        # Skip files in the virtual environment
        if os.path.normpath('venv') in os.path.normpath(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Check if the file needs modification
            if any("sys.path.append" in line for line in lines):
                # Filter out the offending lines and any 'import sys' that might now be unused
                # This is a simple approach; a more complex one would use AST parsing.
                original_line_count = len(lines)
                new_lines = [line for line in lines if "sys.path.append" not in line]
                
                # A basic check to remove 'import sys' if it's the only use
                sys_import_present = any("import sys" in line for line in lines)
                other_sys_usage = any("sys." in line and "sys.path" not in line for line in new_lines)

                if sys_import_present and not other_sys_usage:
                    new_lines = [line for line in new_lines if "import sys" not in line]

                if len(new_lines) < original_line_count:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    
                    print(f"  - Cleaned: {os.path.relpath(file_path, project_dir)}")
                    modified_files_count += 1

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if modified_files_count > 0:
        print(f"\nCleanup complete. Modified {modified_files_count} files.")
    else:
        print("\nNo files containing 'sys.path.append' were found to clean.")

if __name__ == "__main__":
    cleanup_sys_path_hacks()
