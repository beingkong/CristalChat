"""
Main entry point for the voice chatbot application.
"""

import os
import argparse
import logging
from voice_chatbot.orchestrator import OrchestratorModule


def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level (int): Logging level
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Voice Chatbot')
    
    parser.add_argument(
        '--prompt-template',
        type=str,
        help='Path to prompt template file'
    )
    
    parser.add_argument(
        '--voice-ref',
        type=str,
        help='Path to reference voice audio file for TTS'
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
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    # Log startup information
    logging.info("Starting Voice Chatbot")
    logging.info(f"Prompt template: {args.prompt_template}")
    logging.info(f"Voice reference: {args.voice_ref}")
    
    try:
        # Create orchestrator
        orchestrator = OrchestratorModule(
            prompt_template_path=args.prompt_template,
            voice_ref_path=args.voice_ref
        )
        
        # Start voice chatbot
        orchestrator.start()
    
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down")
    
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
    
    finally:
        logging.info("Voice Chatbot stopped")


if __name__ == "__main__":
    main()
