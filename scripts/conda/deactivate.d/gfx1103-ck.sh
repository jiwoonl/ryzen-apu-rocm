if [ "${_ROCM_CK_ACTIVE-}" = 1 ]; then
    if [ "${_ROCM_CK_OLD_LD_SET-}" = x ]; then export LD_LIBRARY_PATH="${_ROCM_CK_OLD_LD-}"; else unset LD_LIBRARY_PATH; fi
    if [ "${_ROCM_CK_OLD_DB_SET-}" = x ]; then export MIOPEN_USER_DB_PATH="${_ROCM_CK_OLD_DB-}"; else unset MIOPEN_USER_DB_PATH; fi
    unset _ROCM_CK_ACTIVE _ROCM_CK_OLD_LD_SET _ROCM_CK_OLD_LD _ROCM_CK_OLD_DB_SET _ROCM_CK_OLD_DB
fi
