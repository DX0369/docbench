from methods.mineru.version_profile import MinerUVersionProfile
from config.method_versions import get_method_version


PROFILE = MinerUVersionProfile(
    method_id="mineru_303_2509",
    display_name="MinerU 3.0.3 + MinerU2.5-2509-1.2B",
    conda_env="gdb_mineru_303",
    repository_relative_path="third_party/methods/mineru_303",
    source_revision=get_method_version("mineru_303_2509").source_revision,
    version="3.0.3",
    checkpoint="OpenDataLab/MinerU2.5-2509-1.2B",
    model_relative_path="mineru/MinerU2.5-2509-1.2B",
    pipeline_model_relative_path="mineru/PDF-Extract-Kit-1.0",
    api_backend="hybrid-auto-engine",
    output_name="mineru_303_2509",
)
