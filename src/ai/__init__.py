"""AI package initialization."""

# AI module initialization - simplified for MVP demo
# Heavy ML dependencies made optional

try:
    from .test_generator import AITestGenerator
    from .rag_system import RAGSystem
    __all__ = ["AITestGenerator", "RAGSystem"]
except ImportError as e:
    # Fallback for missing dependencies
    print(f"Warning: AI dependencies not fully available: {e}")
    __all__ = []
