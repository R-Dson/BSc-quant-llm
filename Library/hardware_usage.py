
from psutil import virtual_memory as vmem
from subprocess import check_output

# This was eventually scrapped and not used.


def get_ram_usage() -> str:
  """
  This function retrieves the amount of RAM (Random Access Memory) being used by the system.

  It utilizes a library called psutil to check the usage and then converts it into gigabytes for readability. The RAM usage is rounded to two decimal places and returned as a string value in GB (gigabytes).

  Returns:
      str: A string representing the amount of RAM being used by the system, in gigabytes.
  """
  try:
    a = vmem().used / (1000 ** 3)  # RAM Used (GB)
    return str(round(a, 2))
  except Exception as e:
     return 'Error reading data'

def get_vram_usage() -> str:
    """
    This function retrieves the amount of Video RAM (VRAM) being used by the GPU.

    It executes a command-line instruction to check the usage and then parses the output
    to extract the relevant information. If it encounters an error while doing so,
    it returns an error message instead. The VRAM usage is returned as a string value in MiB (megabytes).
    Returns:
        str: A string representing the amount of VRAM being used by the GPU, or an error message if there was an issue retrieving the data.
    """
    try:
      vram = check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'])
      vram = vram.decode('utf-8').strip().replace(' MiB', '')
      return vram
    except FileNotFoundError:
      return "Error reading data"

def get_power() -> str:
    """
    This function retrieves the amount of power being drawn by the GPU.

    It executes a command-line instruction to check the power draw and then parses the output
    to extract the relevant information. If it encounters an error while doing so,
    it returns an error message instead. The power usage is returned as a string value in Watts (W).

    Returns:
        str: A string representing the amount of power being drawn by the GPU, or an error message if there was an issue retrieving the data.
    """
    try:
      gpu_output = check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'])
      gpu_output = gpu_output.decode('utf-8').strip()
      return gpu_output.replace(' W\n', '').replace(' W', '')
    except FileNotFoundError:
        return "Error reading data"

