package com.ttennebkram.pipeline.util;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Helper class for running Python ML code with PyTorch.
 * Automatically detects GPU availability (MPS on Mac, CUDA on Windows/Linux).
 *
 * Usage:
 *   PythonMLHelper helper = new PythonMLHelper();
 *   if (!helper.isAvailable()) {
 *       System.out.println(helper.getInstallInstructions());
 *       return;
 *   }
 *   helper.runPythonCode(pythonScript, System.out::println);
 */
public class PythonMLHelper {

    private boolean pythonAvailable = false;
    private boolean pytorchAvailable = false;
    private String pythonVersion = null;
    private String pytorchVersion = null;
    private String deviceType = null;  // "mps", "cuda", or "cpu"
    private String pythonPath = "python3";

    public PythonMLHelper() {
        detectEnvironment();
    }

    /**
     * Check if Python and PyTorch are available.
     */
    public boolean isAvailable() {
        return pythonAvailable && pytorchAvailable;
    }

    public boolean isPythonAvailable() {
        return pythonAvailable;
    }

    public boolean isPyTorchAvailable() {
        return pytorchAvailable;
    }

    public String getPythonVersion() {
        return pythonVersion;
    }

    public String getPyTorchVersion() {
        return pytorchVersion;
    }

    /**
     * Get the detected device type: "mps", "cuda", or "cpu"
     */
    public String getDeviceType() {
        return deviceType;
    }

    /**
     * Get a human-readable description of the GPU status.
     */
    public String getGpuStatus() {
        if (!isAvailable()) {
            return "Not available";
        }
        switch (deviceType) {
            case "mps": return "Apple Silicon GPU (Metal)";
            case "cuda": return "NVIDIA GPU (CUDA)";
            default: return "CPU only (no GPU acceleration)";
        }
    }

    /**
     * Get installation instructions for missing components.
     */
    public String getInstallInstructions() {
        StringBuilder sb = new StringBuilder();
        sb.append("Python ML Setup Required\n");
        sb.append("========================\n\n");

        if (!pythonAvailable) {
            sb.append("Python 3 is not installed or not in PATH.\n\n");
            sb.append("Install Python:\n");
            sb.append("  macOS:   brew install python3\n");
            sb.append("  Windows: Download from https://python.org\n");
            sb.append("  Linux:   sudo apt install python3 python3-pip\n\n");
        }

        if (pythonAvailable && !pytorchAvailable) {
            sb.append("PyTorch is not installed.\n\n");
            sb.append("Install PyTorch:\n");
            sb.append("  pip install torch torchvision\n\n");
            sb.append("Or visit https://pytorch.org for platform-specific instructions.\n");
        }

        return sb.toString();
    }

    /**
     * Attempt to install PyTorch automatically.
     * Returns true if installation succeeded.
     */
    public boolean installPyTorch(Consumer<String> outputHandler) {
        if (!pythonAvailable) {
            outputHandler.accept("Cannot install PyTorch: Python is not available");
            return false;
        }

        outputHandler.accept("Installing PyTorch (this may take a few minutes)...\n");

        try {
            ProcessBuilder pb = new ProcessBuilder(pythonPath, "-m", "pip", "install", "torch", "torchvision");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    outputHandler.accept(line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                outputHandler.accept("\nPyTorch installed successfully!");
                detectEnvironment();  // Re-detect
                return pytorchAvailable;
            } else {
                outputHandler.accept("\nInstallation failed with exit code: " + exitCode);
                return false;
            }
        } catch (Exception e) {
            outputHandler.accept("Installation error: " + e.getMessage());
            return false;
        }
    }

    /**
     * Run Python code and stream output to the handler.
     * Returns the process exit code.
     */
    public int runPythonCode(String pythonCode, Consumer<String> outputHandler) throws IOException, InterruptedException {
        if (!pythonAvailable) {
            throw new IllegalStateException("Python is not available");
        }

        // Write script to temp file
        File tempScript = File.createTempFile("python_ml_", ".py");
        tempScript.deleteOnExit();
        try (PrintWriter writer = new PrintWriter(tempScript)) {
            writer.print(pythonCode);
        }

        // Run Python
        ProcessBuilder pb = new ProcessBuilder(pythonPath, tempScript.getAbsolutePath());
        pb.redirectErrorStream(true);
        pb.environment().put("PYTHONUNBUFFERED", "1");

        Process process = pb.start();

        // Stream output
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                outputHandler.accept(line);
            }
        }

        return process.waitFor();
    }

    /**
     * Run Python code with a timeout.
     * Returns the process exit code, or -1 if timed out.
     */
    public int runPythonCode(String pythonCode, Consumer<String> outputHandler, int timeoutSeconds)
            throws IOException, InterruptedException {
        if (!pythonAvailable) {
            throw new IllegalStateException("Python is not available");
        }

        File tempScript = File.createTempFile("python_ml_", ".py");
        tempScript.deleteOnExit();
        try (PrintWriter writer = new PrintWriter(tempScript)) {
            writer.print(pythonCode);
        }

        ProcessBuilder pb = new ProcessBuilder(pythonPath, tempScript.getAbsolutePath());
        pb.redirectErrorStream(true);
        pb.environment().put("PYTHONUNBUFFERED", "1");

        Process process = pb.start();

        // Stream output in background thread
        Thread outputThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    outputHandler.accept(line);
                }
            } catch (IOException e) {
                // Process ended
            }
        });
        outputThread.start();

        boolean finished = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            outputHandler.accept("\nProcess timed out after " + timeoutSeconds + " seconds");
            return -1;
        }

        outputThread.join(1000);
        return process.exitValue();
    }

    /**
     * Get Python code that sets up device detection.
     * Include this at the start of your ML scripts.
     */
    public static String getDeviceSetupCode() {
        return """
            import torch
            import sys

            # Unbuffered output for real-time streaming
            sys.stdout.reconfigure(line_buffering=True)

            # Auto-detect best available device
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS (Metal) GPU acceleration")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("Using CPU (no GPU acceleration available)")
            """;
    }

    // ========== Private Methods ==========

    private void detectEnvironment() {
        detectPython();
        if (pythonAvailable) {
            detectPyTorch();
        }
    }

    private void detectPython() {
        // Try python3 first, then python
        for (String cmd : new String[]{"python3", "python"}) {
            try {
                ProcessBuilder pb = new ProcessBuilder(cmd, "--version");
                pb.redirectErrorStream(true);
                Process process = pb.start();

                String output = readProcessOutput(process);
                int exitCode = process.waitFor();

                if (exitCode == 0 && output.toLowerCase().contains("python")) {
                    pythonAvailable = true;
                    pythonPath = cmd;
                    pythonVersion = output.trim().replace("Python ", "");
                    return;
                }
            } catch (Exception e) {
                // Try next
            }
        }
        pythonAvailable = false;
    }

    private void detectPyTorch() {
        String checkScript = """
            import torch
            print("VERSION:" + torch.__version__)
            if torch.backends.mps.is_available():
                print("DEVICE:mps")
            elif torch.cuda.is_available():
                print("DEVICE:cuda")
            else:
                print("DEVICE:cpu")
            """;

        try {
            File tempScript = File.createTempFile("check_pytorch_", ".py");
            tempScript.deleteOnExit();
            try (PrintWriter writer = new PrintWriter(tempScript)) {
                writer.print(checkScript);
            }

            ProcessBuilder pb = new ProcessBuilder(pythonPath, tempScript.getAbsolutePath());
            pb.redirectErrorStream(true);
            Process process = pb.start();

            String output = readProcessOutput(process);
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                pytorchAvailable = true;
                for (String line : output.split("\n")) {
                    if (line.startsWith("VERSION:")) {
                        pytorchVersion = line.substring(8).trim();
                    } else if (line.startsWith("DEVICE:")) {
                        deviceType = line.substring(7).trim();
                    }
                }
            }
        } catch (Exception e) {
            pytorchAvailable = false;
        }
    }

    private String readProcessOutput(Process process) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    // ========== Main for testing ==========

    public static void main(String[] args) {
        System.out.println("Python ML Helper - Environment Check");
        System.out.println("=====================================\n");

        PythonMLHelper helper = new PythonMLHelper();

        System.out.println("Python available: " + helper.isPythonAvailable());
        if (helper.isPythonAvailable()) {
            System.out.println("Python version:   " + helper.getPythonVersion());
        }

        System.out.println("PyTorch available: " + helper.isPyTorchAvailable());
        if (helper.isPyTorchAvailable()) {
            System.out.println("PyTorch version:  " + helper.getPyTorchVersion());
            System.out.println("GPU status:       " + helper.getGpuStatus());
        }

        if (!helper.isAvailable()) {
            System.out.println("\n" + helper.getInstallInstructions());
        } else {
            System.out.println("\nReady for ML operations!");

            // Quick test
            System.out.println("\nRunning quick GPU test...\n");
            try {
                String testCode = getDeviceSetupCode() + """

                    # Quick tensor operation test
                    x = torch.randn(1000, 1000, device=device)
                    y = torch.matmul(x, x)
                    print(f"Matrix multiply test passed: {y.shape}")
                    """;

                helper.runPythonCode(testCode, System.out::println);
            } catch (Exception e) {
                System.out.println("Test failed: " + e.getMessage());
            }
        }
    }
}
