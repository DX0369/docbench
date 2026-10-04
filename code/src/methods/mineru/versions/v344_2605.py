from methods.mineru.version_profile import MinerUVersionProfile
from config.method_versions import get_method_version


PROFILE = MinerUVersionProfile(
    method_id="mineru_344_2605",
    display_name="MinerU 3.4.4 + MinerU2.5-Pro-2605-1.2B",
    conda_env="gdb_mineru",
    repository_relative_path="third_party/methods/mineru",
    source_revision=get_method_version("mineru_344_2605").source_revision,
    version="3.4.4",
    checkpoint="OpenDataLab/MinerU2.5-Pro-2605-1.2B",
    model_relative_path="mineru/MinerU2.5-Pro-2605-1.2B",
    pipeline_model_relative_path="mineru/PDF-Extract-Kit-1.0",
    api_backend="hybrid-engine",
    output_name="mineru_344_2605",
)
