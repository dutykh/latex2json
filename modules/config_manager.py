# -*- coding: utf-8 -*-
"""
Configuration manager module for handling settings and API keys.

This module manages loading, validating, and accessing configuration settings
for the chpts2json script.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration settings for the chpts2json script."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "api_keys": {
            "anthropic": "",
            "google": "",
            "google_cx": "",  # Custom Search Engine ID (optional)
            "crossref_email": "",
            "springer": "",
            "elsevier": "",
            "wiley": ""
        },
        "llm_config": {
            "primary_llm": "anthropic",
            "primary_model": "claude-3-5-sonnet-20241022", 
            "fallback_llm": "google",
            "fallback_model": "claude-3-5-sonnet-20241022"
        },
        "preferences": {
            "enable_keyword_generation": True,
            "use_llm_parser": True,
            "llm_parser_cache_days": 30,
            "use_llm_enricher": True,
            "llm_enricher_cache_days": 30,
            "max_retries": 3,
            "timeout": 30,
            "cache_responses": True,
            "cache_expiry_days": 7,
            "delay_between_requests": 1.0
        },
        "scraping": {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "use_selenium": False,
            "selenium_headless": True
        },
        "output": {
            "pretty_print": True,
            "include_metadata": True,
            "sort_by": "year",
            "include_errors": True
        }
    }
    
    def __init__(self, config_file: Optional[Path] = None, verbose: int = 2):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
            verbose: Verbosity level (0-3)
        """
        self.verbose = verbose
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and config_file.exists():
            self._load_config()
        elif config_file:
            if self.verbose >= 1:
                print(f"[CONFIG] Config file not found: {config_file}")
                print("[CONFIG] Using default configuration")
        
        # Override with environment variables if available
        # self._load_env_overrides()  # Not needed for now
    
    def _load_config(self) -> None:
        """Load configuration from file and merge with defaults."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Deep merge user config with defaults
            self.config = self._deep_merge(self.DEFAULT_CONFIG, user_config)
            
            if self.verbose >= 2:
                print(f"[CONFIG] Loaded configuration from {self.config_file}")
            
            # Validate configuration
            self._validate_config()
            
        except json.JSONDecodeError as e:
            if self.verbose >= 1:
                print(f"[CONFIG] Error parsing config file: {e}")
                print("[CONFIG] Using default configuration")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[CONFIG] Error loading config file: {e}")
                print("[CONFIG] Using default configuration")
    
    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Deep merge user configuration with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate numeric values
        prefs = self.config['preferences']
        
        if prefs['max_retries'] < 0:
            prefs['max_retries'] = 0
            if self.verbose >= 1:
                print("[CONFIG] Warning: max_retries must be >= 0, setting to 0")
        
        if prefs['timeout'] < 1:
            prefs['timeout'] = 1
            if self.verbose >= 1:
                print("[CONFIG] Warning: timeout must be >= 1, setting to 1")
        
        if prefs['cache_expiry_days'] < 1:
            prefs['cache_expiry_days'] = 1
            if self.verbose >= 1:
                print("[CONFIG] Warning: cache_expiry_days must be >= 1, setting to 1")
        
        if prefs['delay_between_requests'] < 0:
            prefs['delay_between_requests'] = 0
            if self.verbose >= 1:
                print("[CONFIG] Warning: delay_between_requests must be >= 0, setting to 0")
        
        # Validate sort_by option
        valid_sort_options = ['year', 'title', 'author', 'none']
        if self.config['output']['sort_by'] not in valid_sort_options:
            self.config['output']['sort_by'] = 'year'
            if self.verbose >= 1:
                print("[CONFIG] Warning: invalid sort_by option, setting to 'year'")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "api_keys.anthropic")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def has_api_key(self, service: str) -> bool:
        """Check if an API key is configured for a service."""
        key = self.get(f"api_keys.{service}", "")
        return bool(key and key.strip())
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service."""
        key = self.get(f"api_keys.{service}", "")
        return key if key and key.strip() else None
    
    def save_config(self, output_file: Optional[Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file: Path to save config (uses original file if not specified)
        """
        save_path = output_file or self.config_file
        
        if not save_path:
            if self.verbose >= 1:
                print("[CONFIG] No config file path specified")
            return
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            if self.verbose >= 2:
                print(f"[CONFIG] Saved configuration to {save_path}")
        
        except Exception as e:
            if self.verbose >= 1:
                print(f"[CONFIG] Error saving configuration: {e}")
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for web requests."""
        # Use more browser-like headers to avoid 403 errors
        user_agent = self.get('scraping.user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/json,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        # Hide sensitive API keys
        config_copy = self._deep_merge({}, self.config)
        for key in config_copy.get('api_keys', {}):
            if config_copy['api_keys'][key]:
                config_copy['api_keys'][key] = '***' + config_copy['api_keys'][key][-4:]
        
        return json.dumps(config_copy, indent=2)