from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    
    # OpenAI models available: 
    # gpt-5.2 ($0.00175/input MTok, $0.014/output MTok), 
    # gpt-5.1 ($0.00125/input MTok, $0.01/output MTok), 
    # gpt-4.1-nano ($0.0001/input MTok, $0.0004/output MTok)
    openai_model: str = "gpt-4.1-nano"
    
    # Anthropic models available: 
    # claude-sonnet-4-5 ($0.003/input MTok, $0.015/output MTok), 
    # claude-haiku-4-5 ($0.001/input MTok, $0.005/output MTok), 
    # claude-opus-4-5 ($0.005/input MTok, $0.025/output MTok)
    anthropic_model: str = "claude-haiku-4-5"
    
    max_iterations: int = 5
    agreement_threshold: float = 1.0
    early_stop_threshold: float = 0.90  # Stop at 90% agreement instead of 100%
    
    max_cost_usd: float = 5.0
    
    input_dir: str = "input"
    output_dir: str = "output"
    log_dir: str = "logs"
    
    enable_cache: bool = True
    cache_ttl_hours: int = 24
    
    similarity_threshold: float = 0.85
    numerical_tolerance: float = 0.01
    
    class Config:
        env_file = ".env"

settings = Settings()
