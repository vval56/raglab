import os
import sys
from pathlib import Path

# Make the script runnable directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab import LMStudioLLM

if __name__ == "__main__":
    try:
        answer = LMStudioLLM(
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            model=os.getenv("LMSTUDIO_MODEL", "local-model"),
            timeout=float(os.getenv("LMSTUDIO_TIMEOUT", "120")),
            max_tokens=int(os.getenv("LMSTUDIO_MAX_TOKENS", "16")),
        ).complete("Ответь одним словом: готов")
    except (TimeoutError, OSError, ValueError, KeyError) as exc:
        print(
            "LM Studio недоступен. Запустите локальный сервер, загрузите модель "
            f"и повторите smoke-тест: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    print(answer)
