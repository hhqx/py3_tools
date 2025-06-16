#!/usr/bin/env python
"""
Single Process Debug Example

This script demonstrates how to use the Debugger utility in a single-process context.
It shows how to:
1. Decorate functions with @Debugger.on_error()
2. Trigger debugging through environment variables or command line flags
3. Handle and debug exceptions interactively

Usage:
  # Enable debug via environment variable:
  export IPDB_DEBUG=1
  python debug_single_process.py --mode error

  # Enable debug via command-line flag:
  python debug_single_process.py --mode error --debug
"""

import sys
import argparse
from py_tools.py_debug.debug_utils import Debugger

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Single Process Debugging Example")
    parser.add_argument('--mode', choices=['hello', 'error', 'math_error'], default='hello',
                      help='Execution mode: hello, error, or math_error')
    parser.add_argument('--debug', action='store_true', help='Enable debugging')
    parser.add_argument('--name', type=str, default='Developer', help='Name to greet')
    args = parser.parse_args()
    
    # Enable debug if requested via command line
    if args.debug:
        Debugger.debug_flag = True
        print("Debugging enabled via command line flag")
    elif Debugger.debug_flag:
        print("Debugging enabled via environment variable")
    else:
        print("Debugging disabled. Run with --debug or set IPDB_DEBUG=1")
    
    # Define example functions with debug support
    
    @Debugger.on_error()
    def hello_function(name):
        """Simple greeting function."""
        print(f"Hello, {name}! Debug example is ready.")
        return f"Hello, {name}!"
    
    @Debugger.on_error()
    def error_function():
        """Function that raises a runtime error."""
        print("About to raise an error...")
        raise RuntimeError("This is a test error")
    
    @Debugger.on_error()
    def math_error_function():
        """Function with a division by zero error."""
        print("About to perform division...")
        x = 10
        y = 0
        result = x / y  # This will raise ZeroDivisionError
        return result
    
    # Execute the requested function based on mode
    try:
        if args.mode == 'hello':
            hello_function(args.name)
        elif args.mode == 'error':
            error_function()
        elif args.mode == 'math_error':
            math_error_function()
    except Exception as e:
        print(f"Error caught at top level: {e}")
        sys.exit(1)
    
    print("Program completed successfully")

if __name__ == "__main__":
    main()
