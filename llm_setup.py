"""
llm_setup.py
LLM provider with GPT-4 Turbo
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict

load_dotenv()

class LLMProvider:
    """Flexible LLM provider supporting GPT-4 (extensible to Claude)"""
    
    def __init__(
        self, 
        provider: str = "openai",
        model: str = "gpt-4o"
    ):
        """
        Initialize LLM provider.
        
        Args:
            provider: "openai" or "claude"
            model: Model name
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Provider '{provider}' not yet supported")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            max_tokens: Max response length
            temperature: 0.0 = deterministic, 1.0 = creative
        
        Returns:
            Generated text
        """
        if self.provider == "openai":
            import time
            for attempt in range(6):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        wait = 10 * (attempt + 1)
                        print(f"    [rate limit] waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError("Rate limit retry exhausted")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_with_context(
        self,
        system_prompt: str,
        user_query: str,
        context_chunks: List[Dict],
        max_tokens: int = 500,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response with retrieved context chunks.
        
        Args:
            system_prompt: System instructions
            user_query: User question
            context_chunks: Retrieved chunks from RAG
            max_tokens: Max response length
            temperature: Sampling temperature
        
        Returns:
            Generated response
        """
        # Format context
        context_text = self._format_context(context_chunks)
        
        # Build user prompt with context
        user_prompt = f"""Context from ESG reports and standards:

{context_text}

Question: {user_query}

Please answer based on the provided context. Include chunk IDs as citations."""
        
        return self.generate(system_prompt, user_prompt, max_tokens, temperature)
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks as context"""
        formatted = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Add metadata info
            meta_info = []
            if 'standard_type' in metadata:
                meta_info.append(f"Standard: {metadata['standard_type']}")
            if 'company_id' in metadata:
                meta_info.append(f"Company: {metadata['company_id']}")
            if 'gri_codes' in metadata:
                meta_info.append(f"GRI: {metadata['gri_codes']}")
            
            meta_str = " | ".join(meta_info) if meta_info else ""
            
            formatted.append(f"""[{chunk_id}] {meta_str}
{text}
""")
        
        return "\n---\n".join(formatted)

# Test
if __name__ == "__main__":
    print("="*80)
    print(" LLM PROVIDER TEST")
    print("="*80)
    
    llm = LLMProvider()
    
    # Test 1: Simple generation
    print("\nTest 1: Simple generation")
    print("-"*80)
    
    response = llm.generate(
        system_prompt="You are an ESG compliance expert.",
        user_prompt="What is GRI 305-1 in one sentence?",
        max_tokens=100
    )
    print(response)
    
    # Test 2: With context
    print("\n\nTest 2: Generation with context")
    print("-"*80)
    
    mock_chunks = [
        {
            'chunk_id': 'GRI_305_p0009_c0001',
            'text': 'Disclosure 305-1 requires organizations to report direct (Scope 1) GHG emissions in metric tons of CO2 equivalent.',
            'metadata': {
                'standard_type': 'GRI',
                'gri_codes': ['305-1']
            }
        }
    ]
    
    response = llm.generate_with_context(
        system_prompt="You are an ESG compliance expert. Always cite chunk IDs.",
        user_query="What does GRI 305-1 require?",
        context_chunks=mock_chunks,
        max_tokens=200
    )
    print(response)
    
    print("\n" + "="*80)
    print(" LLM Provider ready!")