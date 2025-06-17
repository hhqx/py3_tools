#!/usr/bin/env python
"""
Single Process Debug Example

This script demonstrates how to use the Debugger utility in a single-process context.
It shows how to:
1. Decorate functions with @Debugger.attach_on_error()
2. Trigger debugging through environment variables or command line flags
3. Handle and debug exceptions interactively
4. Use context manager for debugging specific code blocks

Usage:
  # Enable debug via environment variable:
  export IPDB_DEBUG=1
  python debug_single_process.py --mode error

  # Enable debug via command-line flag:
  python debug_single_process.py --mode error --debug
  
  # Try new context manager example:
  python debug_single_process.py --mode context --debug
  
  # Set logging level:
  python debug_single_process.py --mode error --debug --log_level DEBUG
"""

import sys
import argparse
import logging
from py3_tools.py_debug.debug_utils import Debugger, setup_logging

# Get module logger
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Single Process Debugging Example")
    parser.add_argument('--mode', choices=['hello', 'error', 'math_error', 'context'], default='hello',
                      help='Execution mode: hello, error, math_error, or context')
    parser.add_argument('--debug', action='store_true', help='Enable debugging')
    parser.add_argument('--name', type=str, default='Developer', help='Name to greet')
    parser.add_argument('--debug_mode', choices=['console', 'web', 'socket'], 
                      help='Debug mode (overrides environment variable)')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set logging level')
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    # Enable debug if requested via command line
    if args.debug:
        Debugger.debug_flag = True
        logger.info("Debugging enabled via command line flag")
    elif Debugger.debug_flag:
        logger.info("Debugging enabled via environment variable")
    else:
        logger.info("Debugging disabled. Run with --debug or set IPDB_DEBUG=1")
    
    # Set debug mode if specified
    if args.debug_mode:
        Debugger.debug_mode = args.debug_mode
        logger.info(f"Debug mode set to: {args.debug_mode}")
    
    # Define example functions with debug support
    
    @Debugger.attach_on_error()
    def hello_function(name):
        """Simple greeting function."""
        logger.info(f"Hello, {name}! Debug example is ready.")
        return f"Hello, {name}!"
    
    @Debugger.attach_on_error()
    def error_function():
        """Function that raises a runtime error."""
        logger.warning("About to raise an error...")
        raise RuntimeError("This is a test error")
    
    @Debugger.attach_on_error()
    def math_error_function():
        """Function with a division by zero error."""
        logger.warning("About to perform division by zero...")
        x = 10
        y = 0
        result = x / y  # This will raise ZeroDivisionError
        return result
        
    def context_manager_example():
        """Example using a with-block for debugging a specific code block."""
        logger.info("Starting context manager example...")
        try:
            logger.debug("Before potential error")
            # This block will be debugged if an exception occurs
            risky_operation = lambda: 1/0
            logger.warning("Executing risky operation (division by zero)...")
            risky_operation()
            logger.info("After potential error (this won't be reached)")
        except Exception as e:
            if Debugger.debug_flag:
                logger.error(f"Error in context: {e}")
                frame, tb = sys.exc_info()[1], sys.exc_info()[2]
                if Debugger.debug_mode == 'web':
                    Debugger.web_post_mortem(port=Debugger.base_port)
                else:
                    Debugger.blocking_console_post_mortem(rank=0)
            else:
                raise
    
    # Execute the requested function based on mode
    try:
        logger.debug(f"Executing mode: {args.mode}")
        if args.mode == 'hello':
            hello_function(args.name)
        elif args.mode == 'error':
            error_function()
        elif args.mode == 'math_error':
            math_error_function()
        elif args.mode == 'context':
            context_manager_example()
    except Exception as e:
        logger.critical(f"Error caught at top level: {e}")
        raise
    
    logger.info("Program completed successfully")

if __name__ == "__main__":
    main()
