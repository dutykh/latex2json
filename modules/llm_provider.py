# -*- coding: utf-8 -*-
"""
Unified LLM provider interface for both Anthropic Claude and Google Gemini.

This module provides a consistent interface for LLM operations, allowing
seamless switching between providers based on configuration.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-04
"""

from typing import Optional
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Import provider-specific clients
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """Initialize Anthropic provider."""
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """Generate response using Claude with timeout."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        messages = [{"role": "user", "content": prompt}]
        
        # Use timeout for Claude calls
        timeout_seconds = 30  # 30 second timeout
        
        def _generate():
            return self.client.messages.create(
                model=self.model,
                messages=messages,
                system=system_prompt if system_prompt else "",
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Execute with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            try:
                response = future.result(timeout=timeout_seconds)
                return response.content[0].text
            except TimeoutError:
                raise RuntimeError(f"Claude generation timed out after {timeout_seconds} seconds")
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return ANTHROPIC_AVAILABLE and self.client is not None
    
    def get_model_name(self) -> str:
        """Get the Claude model name."""
        return self.model


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize Gemini provider."""
        self.api_key = api_key
        self.model_name = model
        self.model = None
        
        if GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """Generate response using Gemini with timeout."""
        if not self.model:
            raise RuntimeError("Gemini model not initialized")
        
        # Combine system prompt and user prompt for Gemini
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Use timeout for Gemini calls
        timeout_seconds = 30  # 30 second timeout
        
        def _generate():
            return self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
        
        # Execute with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            try:
                response = future.result(timeout=timeout_seconds)
                return response.text
            except TimeoutError:
                raise RuntimeError(f"Gemini generation timed out after {timeout_seconds} seconds")
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return GEMINI_AVAILABLE and self.model is not None
    
    def get_model_name(self) -> str:
        """Get the Gemini model name."""
        return self.model_name


class LLMProviderManager:
    """Manager for LLM providers with fallback support."""
    
    def __init__(self, config):
        """
        Initialize the provider manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.primary_provider = None
        self.fallback_provider = None
        self.verbose = config.verbose if hasattr(config, 'verbose') else 2
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize primary and fallback providers based on config."""
        # Get LLM configuration
        primary_llm = self.config.get('llm_config.primary_llm', 'anthropic')
        primary_model = self.config.get('llm_config.primary_model', 'claude-3-haiku-20240307')
        fallback_llm = self.config.get('llm_config.fallback_llm', 'google')
        fallback_model = self.config.get('llm_config.fallback_model', 'claude-3-5-sonnet-20241022')
        
        # Initialize primary provider
        if primary_llm == 'anthropic':
            api_key = self.config.get_api_key('anthropic')
            if api_key:
                self.primary_provider = AnthropicProvider(api_key, primary_model)
                if self.verbose >= 2:
                    print(f"[LLM] Primary provider: Anthropic ({primary_model})")
        elif primary_llm == 'google':
            api_key = self.config.get_api_key('google')
            if api_key:
                self.primary_provider = GeminiProvider(api_key, primary_model)
                if self.verbose >= 2:
                    print(f"[LLM] Primary provider: Google Gemini ({primary_model})")
        
        # Initialize fallback provider
        if fallback_llm == 'anthropic':
            api_key = self.config.get_api_key('anthropic')
            if api_key and (not self.primary_provider or primary_llm != 'anthropic'):
                self.fallback_provider = AnthropicProvider(api_key, fallback_model)
                if self.verbose >= 2:
                    print(f"[LLM] Fallback provider: Anthropic ({fallback_model})")
        elif fallback_llm == 'google':
            api_key = self.config.get_api_key('google')
            if api_key and (not self.primary_provider or primary_llm != 'google'):
                self.fallback_provider = GeminiProvider(api_key, fallback_model)
                if self.verbose >= 2:
                    print(f"[LLM] Fallback provider: Google Gemini ({fallback_model})")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: float = 0.3, max_tokens: int = 4096,
                use_fallback: bool = True) -> Optional[str]:
        """
        Generate response using available providers.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            use_fallback: Whether to use fallback provider on failure
            
        Returns:
            Generated text or None if all providers fail
        """
        # Try primary provider
        if self.primary_provider and self.primary_provider.is_available():
            try:
                if self.verbose >= 3:
                    print(f"[LLM] Using primary provider: {self.primary_provider.get_model_name()}")
                return self.primary_provider.generate(
                    prompt, system_prompt, temperature, max_tokens
                )
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[LLM] Primary provider failed: {str(e)}")
                if not use_fallback:
                    return None
        
        # Try fallback provider
        if use_fallback and self.fallback_provider and self.fallback_provider.is_available():
            try:
                if self.verbose >= 2:
                    print(f"[LLM] Using fallback provider: {self.fallback_provider.get_model_name()}")
                return self.fallback_provider.generate(
                    prompt, system_prompt, temperature, max_tokens
                )
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[LLM] Fallback provider failed: {str(e)}")
        
        if self.verbose >= 1:
            print("[LLM] No available providers")
        return None
    
    def is_available(self) -> bool:
        """Check if any provider is available."""
        return ((self.primary_provider and self.primary_provider.is_available()) or
                (self.fallback_provider and self.fallback_provider.is_available()))
    
    def get_active_provider_name(self) -> Optional[str]:
        """Get the name of the active provider."""
        if self.primary_provider and self.primary_provider.is_available():
            return self.primary_provider.get_model_name()
        elif self.fallback_provider and self.fallback_provider.is_available():
            return self.fallback_provider.get_model_name()
        return None