import sys
from unittest.mock import MagicMock

# Create a robust mock that accepts any attribute access under Python 3.12
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

# Now import ragas securely
import ragas
from ragas import evaluate

print("Success! Ragas initialized flawlessly under Python 3.12.")