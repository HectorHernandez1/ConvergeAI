from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    
    max_iterations: int = 5
    agreement_threshold: float = 1.0
    
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
