import os
from pathlib import Path
from unittest.mock import Mock, patch

# Test the exact mock setup from the failing test
def mock_path_side_effect(path_str):
    mock_path = Mock()
    if "existing.py" in str(path_str):
        mock_path.exists.return_value = True
    else:  # deleted.py
        mock_path.exists.return_value = False
    mock_path.is_symlink.return_value = False
    return mock_path

with patch('mcp_server_git.git.operations.Path') as mock_path_class:
    mock_path_class.side_effect = lambda x: mock_path_side_effect(x)
    mock_path_class.return_value.__truediv__.side_effect = mock_path_side_effect
    
    # Test what happens with Path operations
    working_dir = "/test/repo"
    
    print("Testing Path operations...")
    try:
        from mcp_server_git.git.operations import Path as ops_Path
        repo_path = ops_Path(working_dir)
        print(f"repo_path type: {type(repo_path)}")
        
        file_path = repo_path / "existing.py"
        print(f"file_path type: {type(file_path)}")
        print(f"file_path.exists(): {file_path.exists()}")
        
    except Exception as e:
        print(f"Exception during Path operations: {e}")
        
    print("\nTesting os.path fallback...")
    file_path_str = os.path.join(working_dir, "existing.py")
    print(f"os.path.exists('{file_path_str}'): {os.path.exists(file_path_str)}")
