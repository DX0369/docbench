#!/usr/bin/env python3
"""Launch PaddleOCR-VL 1.5/1.6 with normal vLLM execution defaults.

An explicit --enforce-eager is forwarded unchanged. Do not override
torch.compile or silently fall back if the selected execution mode fails.
"""
import uvloop

from vllm.entrypoints.openai.api_server import (
    cli_env_setup,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ModuleNotFoundError:
    from vllm.utils import FlexibleArgumentParser

if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server (PaddleOCR-VL 1.5/1.6)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
