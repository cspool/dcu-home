import opencompass.models.openai_api  # noqa: F401
from opencompass.cli.main import main
from opencompass.datasets import CustomDataset  # noqa: F401
from opencompass.models import OpenAISDK  # noqa: F401
from opencompass.registry import MODELS

MODELS.register_module(module=OpenAISDK, force=True)

if __name__ == "__main__":
    main()
