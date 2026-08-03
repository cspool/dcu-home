# SPDX-License-Identifier: Apache-2.0
"""Fail-closed loader for the frozen gfx936 Qwen3.5 TunableOp profile."""
import hashlib, os
from pathlib import Path
from vllm.logger import init_logger
logger = init_logger(__name__)
_PROFILE_ENV, _SHA_ENV = 'VLLM_ROCM_TUNABLEOP_PROFILE', 'VLLM_ROCM_TUNABLEOP_PROFILE_SHA256'
_PROFILE, _SHA, _VALIDATORS_SHA = 'gfx936_qwen3_5_27b_bf16_tn_m4096', '169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030', '138092e6502e24f4aa30aae012da43451501521e489d9f754ea2df471d30af68'
_VALIDATORS = dict(PT_VERSION='2.10.0', HIP_VERSION='603', GCN_ARCH_NAME='gfx936:sramecc+:xnack-', ROCBLAS_VERSION='4.3.0.ef408db7-dirty', HIPBLASLT_VERSION='1000-a6254b89-dirty')
_SOLUTIONS = {('tn_14336_4096_5120_ld_5120_5120_14336', 'Gemm_Rocblas_20981'), ('tn_16384_4096_5120_ld_5120_5120_16384', 'Gemm_Rocblas_20981'), ('tn_34816_4096_5120_ld_5120_5120_34816', 'Gemm_Rocblas_20981'), ('tn_5120_4096_17408_ld_17408_17408_5120', 'Gemm_Rocblas_20979'), ('tn_5120_4096_6144_ld_6144_6144_5120', 'Gemm_Rocblas_20981')}
_RESULTS = {('GemmTunableOp_BFloat16_TN', p, s, 0.0) for p, s in _SOLUTIONS}
def _fail(message): return RuntimeError(f'ROCm TunableOp profile rejected: {message}')
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _profile_path():
    directory = Path(__file__).with_name('tunable_profiles').resolve(strict=True); candidate = directory / f'{_PROFILE}.csv'
    if candidate.is_symlink(): raise _fail(f'profile path must not be a symlink: {candidate}')
    path = candidate.resolve(strict=True)
    if path.parent != directory or not path.is_file(): raise _fail(f'profile is not a regular wheel-internal file: {path}')
    return path
def _validate_scope(vllm_config):
    import torch
    model, scheduler, parallel = vllm_config.model_config, vllm_config.scheduler_config, vllm_config.parallel_config
    actual = tuple(model.architectures), model.dtype, scheduler.max_num_batched_tokens, parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.prefill_context_parallel_size, parallel.data_parallel_size
    if actual != (('Qwen3_5ForConditionalGeneration',), torch.bfloat16, 4096, 1, 1, 1, 1): raise _fail(f'workload scope mismatch: actual={actual!r}')
    topology = parallel.data_parallel_size_original, parallel.data_parallel_size_local_original, parallel.data_parallel_index, parallel.data_parallel_rank_local
    if topology not in {(1, 1, 0, 0), (2, 2, 0, 0), (2, 2, 1, 1)}: raise _fail(f'workload DP topology mismatch: actual={topology!r}')
    return topology
def _validate_environment(path):
    if (value := os.getenv(_SHA_ENV, '').lower()) != _SHA: raise _fail(f'{_SHA_ENV} mismatch: expected={_SHA}, actual={value!r}')
    allowed = {'PYTORCH_TUNABLEOP_ENABLED': {None, '1'}, 'PYTORCH_TUNABLEOP_TUNING': {None, '0'}, 'PYTORCH_TUNABLEOP_RECORD_UNTUNED': {None, '0'}, 'PYTORCH_TUNABLEOP_FILENAME': {None, str(path)}, 'PYTORCH_TUNABLEOP_ROCBLAS_ENABLED': {None, '1'}, 'PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED': {'0'}}
    conflicts = {k: os.getenv(k) for k, values in allowed.items() if os.getenv(k) not in values}
    if 'PYTORCH_TUNABLEOP_VEROBSE' in os.environ: conflicts['PYTORCH_TUNABLEOP_VEROBSE'] = os.environ['PYTORCH_TUNABLEOP_VEROBSE']
    if conflicts: raise _fail(f'conflicting/cached environment: {conflicts!r}')
def _loaded(path):
    from torch.cuda import tunable
    try: results = tuple((str(o), str(p), str(s), float(t)) for o, p, s, t in tunable.get_results())
    except (TypeError, ValueError) as exc: raise _fail('unexpected get_results payload') from exc
    if len(results) != len(_RESULTS) or set(results) != _RESULTS: raise _fail(f'runtime result mismatch: {results!r}')
    validators = dict(tunable.get_validators())
    if validators != _VALIDATORS: raise _fail(f'runtime validator mismatch: {validators!r}')
    if _sha(path) != _SHA: raise _fail('wheel-internal profile changed after initialization')
def _assert_api_state(path):
    from torch.cuda import tunable
    if not tunable.is_enabled() or tunable.tuning_is_enabled() or tunable.record_untuned_is_enabled(): raise _fail('unexpected TunableOp API state')
    _loaded(path)
def maybe_init_rocm_tunableop(vllm_config, device):
    profile = os.getenv(_PROFILE_ENV, '')
    if not profile:
        logger.info('VLLM_ROCM_TUNABLEOP_INIT status=disabled reason=opt_in_unset')
        return None
    try:
        if profile != _PROFILE: raise _fail(f'unknown profile name: {profile!r}')
        topology, path = _validate_scope(vllm_config), _profile_path()
        _validate_environment(path)
        if _sha(path) != _SHA: raise _fail('profile SHA256 mismatch')
        from torch.cuda import tunable
        tunable.tuning_enable(False); tunable.record_untuned_enable(False)
        tunable.set_filename(str(path), insert_device_ordinal=False); _loaded(path)
        tunable.enable(True); _assert_api_state(path)
        logger.info('VLLM_ROCM_TUNABLEOP_INIT status=ready device=%s file=%s file_sha256=%s validators_sha256=%s rows=5 logical_families=6 explicit_families=5 shared_explicit_families=2 enabled=1 tuning=0 record_untuned=0 parent_dp=%s parent_local_dp=%s dp_index=%s local_dp_rank=%s', device, path, _SHA, _VALIDATORS_SHA, *topology)
        return path
    except Exception as exc:
        logger.exception('VLLM_ROCM_TUNABLEOP_INIT status=error profile=%s reason=%s', profile, exc); raise
def assert_rocm_tunableop_pre_capture(path):
    _assert_api_state(path); logger.info('VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready profile=%s rows=5 expected_hit_keys=5 shared_explicit_families=2 observed_dispatch=external_verbose_canary_required', _PROFILE)
