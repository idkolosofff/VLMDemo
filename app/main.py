from __future__ import annotations

import logging

# Monkey-patch gradio_client to fix "TypeError: argument of type 'bool' is not iterable"
# and "APIInfoParseError: Cannot parse schema True"
# This must happen before any other Gradio imports or usage.
try:
    from gradio_client import utils as _grc_utils
    
    # 1. Patch get_type (for simple cases)
    if hasattr(_grc_utils, "get_type"):
        _orig_get_type = _grc_utils.get_type
        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "bool"
            return _orig_get_type(schema)
        _grc_utils.get_type = _safe_get_type

    # 2. Patch _json_schema_to_python_type (for recursive cases with Pydantic v2)
    # The error happens when 'additionalProperties' is True (boolean) instead of a dict.
    if hasattr(_grc_utils, "_json_schema_to_python_type"):
        _orig_json_to_py = _grc_utils._json_schema_to_python_type
        
        def _safe_json_to_py(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"  # or "dict" or "bool", "Any" is safest for Any/True
            # Also catch the specific recursive call issue within the function if possible,
            # but usually patching the entry point handles the recursive calls if they call the patched function.
            # However, if _json_schema_to_python_type imports itself or is bound, we might need more.
            # Fortunately, in Python modules, internal calls usually resolve to the module-level name.
            return _orig_json_to_py(schema, defs)
            
        _grc_utils._json_schema_to_python_type = _safe_json_to_py
        
except Exception as e:
    print(f"Warning: Failed to patch gradio_client: {e}")


from app.src.config import get_settings
from app.src.services.internvl_service import InternVLService
from app.src.state.session_store import SessionStore
from app.src.ui import build_interface

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def main() -> None:
    settings = get_settings()
    session_store = SessionStore()
    service = InternVLService(settings=settings, session_store=session_store)
    demo = build_interface(settings, service, session_store)
    demo.queue()
    # api_open=False prevents Gradio from crashing when generating API docs for 
    # certain component states (like bare bools in schema).
    demo.launch(
        server_name="0.0.0.0", 
        server_port=settings.port, 
        share=settings.share_ui,
        show_api=False
    )


if __name__ == "__main__":
    main()
