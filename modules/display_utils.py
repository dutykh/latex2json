# -*- coding: utf-8 -*-
"""
Terminal display utilities for beautiful output formatting.

This module provides functions for creating rich, colorful terminal output
with progress indicators, tables, and formatted text.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-04
"""

import sys
from typing import List, Dict, Any
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    RESET = '\033[0m'
    
    @staticmethod
    def disable():
        """Disable colors (for non-terminal output)."""
        for attr in dir(Colors):
            if not attr.startswith('_') and attr != 'disable':
                setattr(Colors, attr, '')


class Icons:
    """Unicode icons for terminal display."""
    CHECK = '✓'
    CROSS = '✗'
    ARROW = '→'
    BULLET = '•'
    STAR = '★'
    INFO = 'ℹ'
    WARNING = '⚠'
    ERROR = '⚡'
    SEARCH = '🔍'
    GLOBE = '🌍'
    PERSON = '👤'
    BRAIN = '🧠'
    SPARKLES = '✨'
    ROCKET = '🚀'
    HOURGLASS = '⏳'
    STOPWATCH = '⏱'
    MAIL = '✉'
    LINK = '🔗'
    BOOK = '📚'
    CHART = '📊'
    PIN = '📍'
    
    # Progress indicators
    SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    PROGRESS = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']


class TerminalDisplay:
    """Enhanced terminal display with colors and formatting."""
    
    def __init__(self, use_colors: bool = True, verbose: int = 2):
        """Initialize display settings."""
        self.use_colors = use_colors and sys.stdout.isatty()
        self.verbose = verbose
        self.spinner_index = 0
        
        if not self.use_colors:
            Colors.disable()
    
    def header(self, text: str, icon: str = Icons.ROCKET):
        """Display a large header."""
        width = 70
        print()
        print(f"{Colors.BRIGHT_CYAN}{'═' * width}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{icon} {Colors.BOLD}{text.center(width - 4)}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'═' * width}{Colors.RESET}")
        print()
    
    def section(self, text: str, icon: str = Icons.ARROW):
        """Display a section header."""
        print()
        print(f"{Colors.BRIGHT_YELLOW}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{icon} {Colors.BOLD}{text}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{'─' * 60}{Colors.RESET}")
    
    def subsection(self, text: str):
        """Display a subsection header."""
        print(f"\n{Colors.CYAN}{Icons.ARROW} {Colors.BOLD}{text}{Colors.RESET}")
    
    def info(self, label: str, value: Any, icon: str = Icons.INFO, color: str = Colors.BLUE):
        """Display an info line."""
        print(f"{color}{icon} {Colors.BOLD}{label}:{Colors.RESET} {value}")
    
    def success(self, text: str):
        """Display a success message."""
        print(f"{Colors.GREEN}{Icons.CHECK} {text}{Colors.RESET}")
    
    def warning(self, text: str):
        """Display a warning message."""
        print(f"{Colors.YELLOW}{Icons.WARNING} {text}{Colors.RESET}")
    
    def error(self, text: str):
        """Display an error message."""
        print(f"{Colors.RED}{Icons.CROSS} {text}{Colors.RESET}")
    
    def progress(self, current: int, total: int, label: str = "Progress"):
        """Display a progress bar."""
        if total == 0:
            return
        
        percent = current / total
        bar_width = 40
        filled = int(bar_width * percent)
        
        # Create progress bar
        bar = Icons.PROGRESS[-1] * filled + Icons.PROGRESS[0] * (bar_width - filled)
        
        # Show progress
        sys.stdout.write(f"\r{Colors.CYAN}{label}: [{bar}] {current}/{total} ({percent*100:.1f}%){Colors.RESET}")
        sys.stdout.flush()
        
        if current == total:
            print()  # New line when complete
    
    def spinner(self, text: str):
        """Display a spinner with text."""
        self.spinner_index = (self.spinner_index + 1) % len(Icons.SPINNER)
        sys.stdout.write(f"\r{Colors.CYAN}{Icons.SPINNER[self.spinner_index]} {text}{Colors.RESET}")
        sys.stdout.flush()
    
    def clear_line(self):
        """Clear the current line."""
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    
    def collaborator_card(self, collab: Dict[str, Any]):
        """Display a formatted collaborator card."""
        print(f"\n{Colors.BRIGHT_MAGENTA}┌{'─' * 58}┐{Colors.RESET}")
        
        # Name and ID
        name = f"{collab.get('firstName', '')} {collab.get('lastName', '')}"
        print(f"{Colors.BRIGHT_MAGENTA}│{Colors.RESET} {Icons.PERSON} {Colors.BOLD}#{collab['id']:03d} {name:<48}{Colors.RESET} {Colors.BRIGHT_MAGENTA}│{Colors.RESET}")
        
        # Separator
        print(f"{Colors.BRIGHT_MAGENTA}├{'─' * 58}┤{Colors.RESET}")
        
        # Details
        details = [
            (Icons.GLOBE, "Affiliation", collab.get('affiliation', 'N/A')),
            (Icons.PIN, "Location", f"{collab.get('city', '')}, {collab.get('country', '')}".strip(', ')),
            (Icons.BRAIN, "Enhanced", "Yes" if collab.get('llm_enhanced') else "No"),
        ]
        
        for icon, label, value in details:
            if value and value != 'N/A':
                formatted_value = value[:45] + '...' if len(str(value)) > 45 else str(value)
                print(f"{Colors.BRIGHT_MAGENTA}│{Colors.RESET} {icon} {Colors.DIM}{label}:{Colors.RESET} {formatted_value:<45} {Colors.BRIGHT_MAGENTA}│{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_MAGENTA}└{'─' * 58}┘{Colors.RESET}")
    
    def search_result_card(self, name: str, results: Dict[str, Any]):
        """Display search results in a formatted card."""
        confidence = results.get('search_confidence', 0.0)
        confidence_color = Colors.GREEN if confidence > 0.7 else Colors.YELLOW if confidence > 0.3 else Colors.RED
        
        print(f"\n{Colors.BRIGHT_BLUE}╭{'─' * 58}╮{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.SEARCH} {Colors.BOLD}Search Results: {name:<35}{Colors.RESET} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}├{'─' * 58}┤{Colors.RESET}")
        
        # Homepage
        if results.get('homepage'):
            print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.LINK} Homepage: {Colors.UNDERLINE}{results['homepage'][:43]}{Colors.RESET} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        
        # Email
        if results.get('email'):
            print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.MAIL} Email: {results['email']:<46} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        
        # Profiles
        profiles = results.get('profiles', {})
        for platform, id_val in profiles.items():
            if id_val:
                platform_display = platform.replace('_', ' ').title()
                print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.STAR} {platform_display}: {str(id_val)[:40]:<40} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        
        # Metrics
        metrics = results.get('academic_metrics', {})
        if any(metrics.values()):
            metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items() if v])[:45]
            print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.CHART} Metrics: {metrics_str:<45} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        
        # Research interests
        interests = results.get('research_interests', [])
        if interests:
            interests_str = ", ".join(interests[:3])[:45]
            print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.BOOK} Interests: {interests_str:<43} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        
        # Confidence
        print(f"{Colors.BRIGHT_BLUE}├{'─' * 58}┤{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {confidence_color}Confidence: {'█' * int(confidence * 10)}{'░' * (10 - int(confidence * 10))} {confidence:.1%}{Colors.RESET}{' ' * 30} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_BLUE}╰{'─' * 58}╯{Colors.RESET}")
    
    def summary_table(self, stats: Dict[str, Any]):
        """Display a summary table."""
        print(f"\n{Colors.BRIGHT_GREEN}┌{'─' * 40}┬{'─' * 17}┐{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}│{Colors.BOLD} {'Metric':<38} {Colors.RESET}{Colors.BRIGHT_GREEN}│{Colors.BOLD} {'Value':>15} {Colors.RESET}{Colors.BRIGHT_GREEN}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}├{'─' * 40}┼{'─' * 17}┤{Colors.RESET}")
        
        for key, value in stats.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            print(f"{Colors.BRIGHT_GREEN}│{Colors.RESET} {key:<38} {Colors.BRIGHT_GREEN}│{Colors.RESET} {value_str:>15} {Colors.BRIGHT_GREEN}│{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_GREEN}└{'─' * 40}┴{'─' * 17}┘{Colors.RESET}")
    
    def feature_status(self, features: List[tuple]):
        """Display feature availability status."""
        print()
        for available, name, details in features:
            if available:
                print(f"{Colors.GREEN}{Icons.CHECK} {name}: {Colors.BOLD}{details}{Colors.RESET}")
            else:
                print(f"{Colors.RED}{Icons.CROSS} {name}: {Colors.DIM}Not configured{Colors.RESET}")
    
    def timestamp(self):
        """Return formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S")
    
    def timed_info(self, text: str, icon: str = Icons.INFO):
        """Display info with timestamp."""
        print(f"{Colors.DIM}[{self.timestamp()}]{Colors.RESET} {icon} {text}")
    
    def detailed_extraction_info(self, entry_num: int, total: int, raw_entry: str, extracted_data: Dict[str, Any]):
        """Display detailed extraction information for verbosity level 3."""
        # Header with entry number
        print(f"\n{Colors.BRIGHT_CYAN}╭{'─' * 70}╮{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET} {Icons.PERSON} {Colors.BOLD}Processing Entry #{entry_num}/{total}{Colors.RESET}{' ' * (70 - 25 - len(str(entry_num)) - len(str(total)))} {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}├{'─' * 70}┤{Colors.RESET}")
        
        # Raw LaTeX entry (truncated if too long)
        raw_display = raw_entry.strip()
        if len(raw_display) > 65:
            raw_display = raw_display[:62] + "..."
        print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET} {Icons.BOOK} Raw: {raw_display:<63} {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}├{'─' * 70}┤{Colors.RESET}")
        
        # Extracted fields
        fields = [
            ("Name", f"{extracted_data.get('firstName', '')} {extracted_data.get('lastName', '')}".strip()),
            ("Affiliation", extracted_data.get('affiliation', '')),
            ("University", extracted_data.get('university', '')),
            ("Department", extracted_data.get('department', '')),
            ("City", extracted_data.get('city', '')),
            ("Country", extracted_data.get('country', '')),
        ]
        
        for label, value in fields:
            if value:
                value_display = str(value)[:54] if len(str(value)) > 54 else str(value)
                print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}   {Colors.DIM}{label:12}{Colors.RESET} {value_display:<54} {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        
        # Enhanced location if available
        if extracted_data.get('enhanced_location'):
            print(f"{Colors.BRIGHT_CYAN}├{'─' * 70}┤{Colors.RESET}")
            loc = extracted_data['enhanced_location']
            if len(loc) > 52:
                loc = loc[:49] + "..."
            confidence = extracted_data.get('location_confidence', 0.0)
            print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET} {Icons.SPARKLES} Enhanced: {loc:<52} ({confidence:.0%}) {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        
        # LLM status
        if extracted_data.get('llm_enhanced'):
            print(f"{Colors.BRIGHT_CYAN}├{'─' * 70}┤{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET} {Icons.BRAIN} {Colors.GREEN}LLM Enhanced ✓{Colors.RESET}{' ' * 54} {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_CYAN}╰{'─' * 70}╯{Colors.RESET}")
    
    def extraction_field(self, label: str, value: str, indent: int = 2):
        """Display a single extracted field with proper formatting."""
        if value:
            indent_str = " " * indent
            print(f"{indent_str}{Colors.DIM}{label}:{Colors.RESET} {value}")
    
    def batch_processing_header(self, batch_num: int, total_batches: int, start_idx: int, end_idx: int):
        """Display header for batch processing."""
        print(f"\n{Colors.BRIGHT_YELLOW}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Icons.BRAIN} Batch {batch_num}/{total_batches} (Entries {start_idx}-{end_idx}){Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{'═' * 60}{Colors.RESET}\n")
    
    def search_attempt_detail(self, name: str, affiliation: str, country: str, search_num: int, total: int):
        """Display detailed search attempt information."""
        print(f"\n{Colors.BRIGHT_BLUE}┌{'─' * 60}┐{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET} {Icons.SEARCH} Search #{search_num}/{total}: {name:<41} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        if affiliation:
            aff_display = affiliation[:54] if len(affiliation) > 54 else affiliation
            print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET}   🏢 {aff_display:<54} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        if country:
            print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET}   🌍 {country:<54} {Colors.BRIGHT_BLUE}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}└{'─' * 60}┘{Colors.RESET}")
    
    def llm_processing_detail(self, action: str, details: str = None):
        """Display LLM processing details."""
        print(f"      {Icons.BRAIN} {Colors.CYAN}{action}{Colors.RESET}")
        if details:
            print(f"         {Colors.DIM}{details}{Colors.RESET}")