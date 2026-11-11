if [ "${_NPU_XRT_PATH_ADDED-0}" = 1 ]; then
    case "$PATH" in *:"$HOME/.local/opt/xrt/bin") export PATH="${PATH%:$HOME/.local/opt/xrt/bin}" ;; esac
fi
if [ "${_NPU_XRT_PREV_SET-}" = x ]; then export XILINX_XRT="${_NPU_XRT_PREV-}"; else unset XILINX_XRT; fi
unset _NPU_XRT_PREV_SET _NPU_XRT_PREV _NPU_XRT_PATH_ADDED
