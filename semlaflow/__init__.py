# Must happen before torch is ever imported, in any script under this package. On aarch64,
# torch's CPU matmul dispatch routes through oneDNN to ARM Compute Library for GEMM, which sizes
# its own OpenMP thread pool from these env vars at library load / first use -- independently of
# torch.set_num_threads(), which does NOT reach it (confirmed from a crash where GOMP_parallel
# was still requesting the full core count despite torch.set_num_threads(1) being set in a
# DataLoader worker_init_fn). libgomp is not fork-safe: a forked DataLoader worker building a
# >1-thread OpenMP team for a matmul (eg. GeometricMol.soft_permute's P @ coords / einsum, used
# by sinkhorn coupling) segfaults. Single-threaded is the only safe setting for these libraries
# in this process, set as early as physically possible since scripts run via `python -m
# semlaflow.<script>`, which imports this file before the script's own body.
import os

for _omp_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_omp_var, "1")

del _omp_var
