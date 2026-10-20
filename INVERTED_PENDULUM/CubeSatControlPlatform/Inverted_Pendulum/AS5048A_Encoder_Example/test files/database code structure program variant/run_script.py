import subprocess

# Use xterm as an alternative
process1 = subprocess.Popen(['xterm', '-e', 'python3', 'encoderbestest.py'])
process2 = subprocess.Popen(['xterm', '-e', 'python3', 'extension.py'])

process1.wait()
process2.wait()
