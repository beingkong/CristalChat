"""
Local version of the voice chatbot that uses text input/output instead of audio.
"""

import os
import argparse
import logging
from voice_chatbot.orchestrator import OrchestratorModule


def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Local Voice Chatbot')
    
    parser.add_argument(
        '--prompt-template',
        type=str,
        help='Path to prompt template file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the local application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    # Log startup information
    logging.info("Starting Local Voice Chatbot")
    logging.info(f"Prompt template: {args.prompt_template}")
    
    try:
        # Create orchestrator
        orchestrator = OrchestratorModule(
            prompt_template_path=args.prompt_template,
            use_local_mode=True  # Add this flag to indicate local mode
        )
        
        # Start chat loop
        print("\nWelcome to the Local Voice Chatbot!")
        print("Type your message and press Enter to send.")
        print("Type 'quit' or press Ctrl+C to exit.\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Process user input
            response = orchestrator.process_text_input(user_input)
            
            # Display response
            print(f"Bot: {response}\n")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
    finally:
        logging.info("Local Voice Chatbot stopped")


if __name__ == "__main__":
    main()
