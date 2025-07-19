#!/usr/bin/env python3
"""
Minimal debug test for MCP list_tools issue
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

async def debug_mcp_server():
    """Debug the MCP server list_tools issue"""
    print("🔍 Starting MCP server debug test...")
    
    # Start server process
    import os
    env = os.environ.copy()
    env.update({"GITHUB_TOKEN": "test_token", "PYTHONPATH": "src"})
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "mcp_server_git",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=Path.cwd(),
    )
    
    try:
        # Send initialize request first
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"experimental": {}, "sampling": {}},
                "clientInfo": {"name": "debug-client", "version": "1.0.0"},
            },
        }
        
        init_json = json.dumps(init_request) + "\n"
        process.stdin.write(init_json.encode())
        await process.stdin.drain()
        
        # Read initialize response
        init_response_line = await process.stdout.readline()
        if init_response_line:
            init_response = json.loads(init_response_line.decode())
            print(f"✅ Initialize response: {init_response}")
        else:
            print("❌ No initialize response received")
            return
        
        # Send tools/list request WITHOUT params
        tools_request_no_params = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        print(f"📤 Sending tools/list request (no params): {tools_request_no_params}")
        tools_json = json.dumps(tools_request_no_params) + "\n"
        process.stdin.write(tools_json.encode())
        await process.stdin.drain()
        
        # Read tools response
        tools_response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        if tools_response_line:
            tools_response = json.loads(tools_response_line.decode())
            print(f"📥 Tools response (no params): {tools_response}")
        else:
            print("❌ No tools response received (no params)")
        
        # Send tools/list request WITH empty params
        tools_request_with_params = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {}
        }
        
        print(f"📤 Sending tools/list request (with params): {tools_request_with_params}")
        tools_json = json.dumps(tools_request_with_params) + "\n"
        process.stdin.write(tools_json.encode())
        await process.stdin.drain()
        
        # Read tools response
        tools_response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        if tools_response_line:
            tools_response = json.loads(tools_response_line.decode())
            print(f"📥 Tools response (with params): {tools_response}")
        else:
            print("❌ No tools response received (with params)")
            
        # Test prompts/list to see if other handlers work
        prompts_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/list",
            "params": {}
        }
        
        print(f"📤 Sending prompts/list request: {prompts_request}")
        prompts_json = json.dumps(prompts_request) + "\n"
        process.stdin.write(prompts_json.encode())
        await process.stdin.drain()
        
        # Read prompts response
        prompts_response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        if prompts_response_line:
            prompts_response = json.loads(prompts_response_line.decode())
            print(f"📥 Prompts response: {prompts_response}")
        else:
            print("❌ No prompts response received")
            
    except asyncio.TimeoutError:
        print("⏰ Timeout waiting for server response")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Check stderr for any error messages
        try:
            stderr_output = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
            if stderr_output:
                print(f"🔍 Server stderr: {stderr_output.decode()}")
        except asyncio.TimeoutError:
            pass
        
        # Cleanup
        process.terminate()
        await process.wait()

if __name__ == "__main__":
    asyncio.run(debug_mcp_server())