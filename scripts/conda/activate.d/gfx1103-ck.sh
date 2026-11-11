# Enable the locally built ROCm 10.1.0 gfx1103 MIOpen plugin in this environment.
if [ -f "$HOME/.local/lib/rocm-gfx1103-ck/libMIOpenCKGroupedConv_gfx1103.so" ] && [ -d "$CONDA_PREFIX/lib/python3.13/site-packages/rocm_sdk_libraries-10.1.0.dist-info" ]; then
    export _ROCM_CK_OLD_LD_SET="${LD_LIBRARY_PATH+x}"
    export _ROCM_CK_OLD_LD="${LD_LIBRARY_PATH-}"
    export _ROCM_CK_OLD_DB_SET="${MIOPEN_USER_DB_PATH+x}"
    export _ROCM_CK_OLD_DB="${MIOPEN_USER_DB_PATH-}"
    export LD_LIBRARY_PATH="$HOME/.local/lib/rocm-gfx1103-ck${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export MIOPEN_USER_DB_PATH="$HOME/.cache/rocm-gfx1103-ck/miopen"
    mkdir -p "$MIOPEN_USER_DB_PATH"
    export _ROCM_CK_ACTIVE=1
else
    echo 'ROCm gfx1103 CK plugin not enabled: library missing or ROCm package version changed.' >&2
fi
