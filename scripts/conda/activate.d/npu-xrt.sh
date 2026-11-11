# NPU XRT tools for the active conda environment; retain the caller's previous setting.
export _NPU_XRT_PREV_SET="${XILINX_XRT+x}"
export _NPU_XRT_PREV="${XILINX_XRT-}"
export XILINX_XRT="$HOME/.local/opt/xrt"
export _NPU_XRT_PATH_ADDED=0
case ":$PATH:" in
  *":$XILINX_XRT/bin:"*) ;;
  *) export PATH="$PATH:$XILINX_XRT/bin"; export _NPU_XRT_PATH_ADDED=1 ;;
esac
