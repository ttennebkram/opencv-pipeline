# OpenCV Pipeline Editor

A graphical node-based pipeline editor for OpenCV image processing operations.

**IMPORTANT: Always read README.md as well when starting a session.** The README contains user-facing documentation including command line options, feature descriptions, and platform support details that provide important context.

## IMPORTANT: Git Policy

**DO NOT create git commits or push without explicit user approval.**

- You may SUGGEST a commit when work is complete
- Wait for the user to test and confirm before committing
- Never commit code in a potentially broken state
- The user prefers to verify changes work before committing

## IMPORTANT: Verification

**Do not claim a fix works until it has been tested.**

- After making changes, compile with `mvn compile` to catch errors
- Do not say "this should fix it" or "this is now fixed" - say "I've made changes, please test"
- If you're unsure whether something works, say so
- The user will verify functionality before considering something complete

## Build & Run

```bash
# Compile
mvn compile

# Run (normal)
mvn exec:exec

# Run with auto-start (loads last file and starts pipeline)
mvn exec:exec@start

# Run with custom arguments
mvn exec:exec@run -Dpipeline.args="file.json -a"
mvn exec:exec@run -Dpipeline.args="file.json -a --fullscreen_node_name Monitor"
mvn exec:exec@run -Dpipeline.args="file.json -a --camera_fps 15 --max_time 60"

# Package uber-jar
mvn clean package
java -jar target/opencv-pipeline.jar

# Build all platform JARs for release
./build-all-platforms.sh
# Creates: target/opencv-pipeline-{mac-aarch64,mac,linux,linux-aarch64,win}.jar
```

## Tech Stack

- **Java 17** with **JavaFX 23** for UI
- **OpenCV 4.9.0** via openpnp wrapper (includes native libs for all platforms)
- **Gson** for JSON serialization
- **Maven** build system

## Project Structure

```
src/main/java/com/ttennebkram/pipeline/
├── PipelineEditorApp.java       # Main application, toolbar, canvas
├── PipelineEditorLauncher.java  # Entry point (for JavaFX module workaround)
├── fx/
│   ├── FXNode.java              # Visual node representation
│   ├── FXConnection.java        # Connection between nodes
│   ├── FXPipelineExecutor.java  # Runs the pipeline, manages Mat objects
│   ├── FXPipelineSerializer.java # Save/load JSON, thumbnails
│   ├── FXNodeRegistry.java      # Discovers and lists available node types
│   ├── FXNodeFactory.java       # Creates FXNode instances
│   ├── FXPropertiesDialog.java  # Node property editor dialogs
│   ├── FXContainerEditorWindow.java # Editor for Container sub-pipelines
│   ├── FXHelpBrowser.java       # Help documentation browser
│   ├── NodeRenderer.java        # Renders nodes on canvas
│   └── processors/              # All image processing nodes
│       ├── FXProcessor.java         # Base interface
│       ├── FXProcessorBase.java     # Common implementation
│       ├── FXProcessorInfo.java     # Annotation for auto-discovery
│       ├── FXProcessorRegistry.java # Maps nodeType -> processor class
│       ├── FXDualInputProcessor.java # Base for dual-input nodes
│       ├── FXMultiOutputProcessor.java # Base for multi-output nodes
│       └── *Processor.java          # Individual processor implementations
├── processing/                  # Legacy processor interfaces
└── util/
    └── MatTracker.java          # Tracks OpenCV Mat allocations for leak detection
```

## Adding a New Processor

1. Create a class in `fx/processors/` extending `FXProcessorBase` (or `FXDualInputProcessor` for two inputs, `FXMultiOutputProcessor` for multiple outputs)

2. Add the `@FXProcessorInfo` annotation:
```java
@FXProcessorInfo(
    nodeType = "MyFilter",           // Internal name (no spaces)
    displayName = "My Filter",        // Shown in UI
    buttonName = "MyFilt",            // Optional: shorter toolbar name
    category = "Filter",              // Category in toolbar
    description = "Does something\nSecond line"
)
public class MyFilterProcessor extends FXProcessorBase {
```

3. Implement required methods:
   - `process(Mat input)` - the actual image processing
   - `buildPropertiesDialog(FXPropertiesDialog dialog)` - UI for parameters
   - `serializeProperties(JsonObject json)` / `deserializeProperties(JsonObject json)`
   - `syncFromFXNode(FXNode node)` / `syncToFXNode(FXNode node)`

4. The processor is auto-discovered via annotation scanning (no registration needed)

## Categories

Toolbar categories appear in this order (defined in `FXNodeRegistry.CATEGORY_ORDER`):
1. Sources
2. Basic
3. Blur
4. Content
5. Edges
6. Filter
7. Morphology
8. Transform
9. Detection
10. Dual Input
11. Utility
12. Visualization
13. Container I/O

## Key Patterns

### Node Status Text
Processors can display extra runtime information (like computed values) via `FXNode.statusText`:
```java
// In process() method, update statusText for display on the node
if (fxNode != null) {
    fxNode.statusText = String.format("Value: %.0f", computedValue);
}
```
- statusText is displayed after "Work:" in the node's stats line
- Cleared when pipeline starts (in `clearPipelineStats()`)
- Persisted to JSON and restored on load

### Mat Memory Management
OpenCV Mats must be manually released. Use `MatTracker` for debugging leaks:
```java
MatTracker.register(mat, "description");
MatTracker.release(mat);
MatTracker.dumpStats();  // Print allocation info
```

### Properties Dialog Helpers
`FXPropertiesDialog` provides common controls:
- `addSlider(label, min, max, value)`
- `addOddKernelSlider(label, value, max)` - for odd-only kernel sizes
- `addCheckBox(label, checked)`
- `addComboBox(label, options, selected)`
- `addColorPicker(label, color)`

### Serialization Helpers in FXProcessorBase
```java
getJsonInt(json, "key", defaultVal)
getJsonDouble(json, "key", defaultVal)
getJsonString(json, "key", defaultVal)
getInt(properties, "key", defaultVal)  // from FXNode.properties map
```

## File Format

Pipelines are saved as JSON with `.json` extension. Thumbnails are cached in `~/.opencv-pipeline/thumbnails/`.

## Raspberry Pi Camera Support

The webcam source supports Raspberry Pi cameras (CSI modules like OV5647) via `rpicam-still` polling.

### How it works

On Raspberry Pi 5, the camera uses libcamera which is not compatible with OpenCV's V4L2 backend. OpenCV can *open* the V4L2 device but cannot *read* frames from it. The solution:

1. `FXWebcamSource.open()` first checks if `rpicam-still` is available
2. If found, it starts `rpicam-still --timelapse 100` which writes JPEG frames to a temp file every 100ms
3. `captureFrame()` reads the latest JPEG from the temp file using `Imgcodecs.imread()`
4. Falls back to standard V4L2 VideoCapture for USB webcams

### Key files

- `FXWebcamSource.java` - `tryOpenWithRpicam()`, `captureFrameFromRpicam()`, `stopRpicamProcess()`

### Alternatives considered but not viable

- **GStreamer pipeline**: OpenCV (openpnp build) is not compiled with GStreamer support
- **rpicam-vid TCP/UDP streaming**: OpenCV couldn't read the streams reliably
- **Building custom OpenCV**: Would break cross-platform portability

### USB webcams

USB webcams work normally via V4L2 and don't need the rpicam workaround. The code automatically detects which method to use.

## Command Line Options

Key options for automated/scripted use:
- `-a`, `--auto_start`, `--auto_run` - Start pipeline immediately after loading
- `--fullscreen_node_name NAME` - Open fullscreen preview of named node (live, updates continuously)
- `--max_time SECONDS` - Exit after specified time (timer thread is daemon, so app exits cleanly)
- `--camera_index`, `--camera_resolution`, `--camera_fps`, `--camera_mirror` - Override all webcam sources

Options can appear before or after the pipeline filename.

## Known Issues

- Watch for file handle leaks - always close InputStreams, use try-with-resources
- FFT nodes have complex multi-channel output modes (grayscale vs 4-channel)
- Container nodes have their own sub-pipeline serialized inline

## Recent Bug Fixes Worth Noting

- **Node label serialization** (v2.4.0): Custom labels that match the node type name (e.g., naming a Monitor node "Monitor") now save/restore correctly. Previously the condition `!savedLabel.equals(type)` rejected these.
- **Fullscreen preview race conditions**: The fullscreen window uses an `AnimationTimer` to continuously update from `node.previewImage`, and properly requests focus to avoid beeps on first click.
- **Clean exit with --max_time**: The timer thread is set as daemon (`setDaemon(true)`) so the JVM exits when windows close.
- **Pipeline stop race condition**: Null checks added for `pipelineExecutor` in callbacks that may fire after pipeline stops.

---

## Claude Code Wrapper (ccwrap)

**Note:** `DIALOG.md` exists in the `_dialog` branch but may not be visible on this branch.

### Goal

A Python 3 wrapper that runs Claude Code underneath and adds:

1. **Journaling**: Write an auditable dialog log to `DIALOG.md` in the repo root with:
   - A fixed "Latest Dialog" block at the top that is rewritten each update (do not rely on anchors)
   - Append-only entries below that clearly separate User vs Claude
   - Timestamps including timezone

2. **Awaiting-user detection**:
   - Detect when Claude Code is waiting for input (regex + heuristics)
   - When detected, send email to `mbennett+cc@ideaeng.com` with a short per-session token and prompt context
   - While in awaiting mode, poll Gmail label `CC` for replies containing the token and a command (y/n/1-9/status/abort)
   - If accepted, inject that reply into Claude Code stdin

3. **Dialog branch updates**:
   - Update branch `_dialog` without checkout so `_dialog` contains only `DIALOG.md`
   - Use Git plumbing: `hash-object`, `mktree`, `commit-tree`, `update-ref`
   - Do not modify the working tree or current branch
   - Do not require hooks or worktrees

### Package Location

```
tools/ccwrap/
├── __init__.py
├── __main__.py
├── main.py
├── pty_runner.py
├── dialog_log.py
├── git_dialog_branch.py
├── gmail_client.py
├── email_notify.py
├── config.py
└── util.py
```

Runnable entrypoint: `python3 -m tools.ccwrap`

### Running Claude Code

```bash
python3 -m tools.ccwrap [-- <args passed to claude-code>]
```

The wrapper spawns `claude` or `claude-code` as a child process in a PTY (pseudo-terminal) so prompts behave as in a real terminal. Captures all child stdout/stderr (PTY stream) and all user keystrokes sent to child.

### DIALOG.md Format

At top of `DIALOG.md`, maintain a fixed block delimited by sentinels:

```markdown
<!-- LATEST-START -->
## Latest Dialog
Updated: YYYY-MM-DD HH:MM:SS ±HHMM

Scroll down to see the most recent full entry below.
<!-- LATEST-END -->

---
```

Below that, append entries like:

```markdown
---
## YYYY-MM-DD HH:MM:SS ±HHMM

**Session ID:** <uuid>
**Repo:** <name>
**Branch:** <branch>
**HEAD:** <sha>
**Host:** <hostname>

### User
```text
<exact user input (raw)>
```

### Claude
<relevant Claude output since last user input>
```

Rules:
- Rewrite only the Latest block
- Append entries; never rewrite history
- Clearly label User vs Claude
- Use local timezone

### Awaiting-User Detection

Triggers awaiting state if PTY output matches any of these patterns (case-insensitive):
- `\(y/n\)` or `\[(y/n)\]`
- `continue\?`
- `press enter`
- `choose (one|an option)`
- `select (1|one)`
- `are you sure`
- `confirm`
- `waiting for (your )?input`

Idle heuristic: If no PTY output for N seconds AND the last line ends with `:` or `?` or `]` or `)`, assume awaiting input.

When awaiting is triggered:
- Generate a short token (e.g., 6 chars A-Z0-9)
- Send email with subject: `[CC][WAITING][TOKEN] <repo> (<branch>) — needs input`
- Include prompt excerpt (last ~30 lines) and instructions
- Rate-limit: do not spam more than once per awaiting event

### Gmail Polling and Reply Injection

Use Gmail API (not IMAP/POP3).
- Query: `label:CC is:unread`
- Accept reply only if:
  - From address is in allowlist
  - Contains current token
  - Command grammar matches: `y` / `n` / digits `1`-`9` / `status` / `abort`
- When accepted:
  - Write command + newline into PTY child stdin
  - Mark email processed by removing `UNREAD` label
  - Exit awaiting mode

### Git `_dialog` Update (No Checkout)

Function `update_dialog_branch(dialog_path="DIALOG.md", branch="_dialog")`:
1. Compute blob hash of file: `git hash-object -w --stdin`
2. If `_dialog` exists and `rev-parse _dialog:DIALOG.md` equals new blob, do nothing
3. Create one-file tree via `git mktree`
4. Create commit via `git commit-tree` with parent if exists
5. `git update-ref refs/heads/_dialog <commit>`

This branch tree contains only `DIALOG.md`.

### Configuration

`tools/ccwrap/config.py` settings:
- Gmail credentials path / token path
- Label name: `CC`
- To address: `mbennett+cc@ideaeng.com`
- Allowlist senders
- Polling interval (e.g., 10s)
- Awaiting idle threshold (e.g., 15s)

Support environment overrides.

### OAuth Setup

`tools/ccwrap/gmail_client.py` implements Gmail API auth for installed apps:
- Store token locally (e.g., `.ccwrap_token.json` in repo root or `~/.config/ccwrap/`)
- Use Gmail scope `gmail.modify`
- Provide `--gmail-auth` mode that runs the OAuth flow once

### Event Logging

Write a JSONL event log (optional file `ccwrap_events.jsonl` in repo root):
- session_start/end
- stdout chunk
- user_input
- awaiting_enter/exit
- email_sent
- gmail_poll
- reply_accepted/rejected
- git_dialog_updated

---

## ML Experiments: Corner Detection CNN

### Progressive Overfit Testing (√2 Scaling)

Validates the CNN can memorize increasingly larger datasets before training on full data.

**Method:** Scale dataset by √2 (~1.41×) each step, require <5px error before advancing.

**Architecture:** ResNet-style CNN with BatchNorm
- 4 ResBlocks: 1→32→64→128→256 channels
- Input: 240×135 grayscale (1/8 of 1920×1080)
- Output: 8 corner coordinates (normalized -1 to 1)
- ~4.9MB checkpoint size

**Training Settings:**
- Optimizer: Adam, initial LR: 0.0005
- Scheduler: ReduceLROnPlateau(patience=50, factor=0.5)
- Batch size: 256
- Warm start from 500-sample checkpoint

**Results (2025-12-16):**

| Samples | Best Error | Epochs | Status |
|---------|------------|--------|--------|
| 50 | 4.5px | 149 | PASS |
| 500 | 4.6px | 430 | PASS |
| 707 | 5.0px | 302 | PASS |
| 1000 | 4.3px | 303 | PASS |
| 1414 | 4.1px | 308 | PASS |
| 2000 | 4.1px | 335 | PASS |
| 2828 | 4.7px | 641 | PASS |
| 4000 | 4.7px | 303 | PASS |
| 5656 | 4.8px | 299 | PASS |
| 8000 | 4.6px | 267 | PASS |
| 11314 | 5.0px | 1040 | PASS |

**Next:** 16000 samples (11314 × √2)

### Data Generation

**Local:** 1M+ images generated in `/Volumes/SamsungBlue/ml-training/wiki_training_v3/`

**AWS:**
- Instance: **DELETED** (was g5.xlarge with A10G GPU)
- S3 only: `s3://ml-training-wiki-homography/wiki_training_v3/` (~$2-3/month)
- Original JPEGs + labels backed up in S3

**To recreate AWS training environment:**
1. Launch new g5.xlarge with Deep Learning AMI
2. Install dependencies: `pip3 install torch onnx`
3. Sync training data: `aws s3 sync s3://ml-training-wiki-homography/wiki_training_v3/ /opt/dlami/nvme/wiki_training_v3/`
4. Convert JPEGs to tensors (see conversion script below) - takes ~2 hours for 1M images
5. Copy checkpoint from local: `scp models/full_training_24.3px_epoch20.pth ubuntu@<ip>:/opt/dlami/nvme/`

### Key Files

- `experiments/generate_wiki_training_css3d.py` - CSS 3D transforms in browser
- `experiments/train_corners_cnn.py` - CNN training script
- `/Volumes/SamsungBlue/ml-training/progressive_checkpoints/` - All checkpoints

### Full-Scale Training Optimization (2025-12-17)

#### Data Loading Bottleneck

Training on 1M+ images was bottlenecked by JPEG decoding, not GPU compute:
- JPEG decode: ~5-10ms per image (OpenCV imread + resize)
- GPU batch processing: ~50ms for batch of 256
- Result: GPU utilization only 20-40%

#### Solution: Pre-convert to PyTorch Tensors

Convert JPEGs to `.pt` tensor files (uint8, already resized to 240×135):
```python
# Conversion script creates tensors ~34KB each (vs 190-280KB JPEGs)
img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (240, 135))
torch.save(torch.from_numpy(img), tensor_path)
```

Results:
- Tensor load: ~1ms (vs 5-10ms for JPEG)
- GPU utilization: 100%
- Epoch time: 4× faster

**Data Quality Issues Found:**
- 1,056,312 original JPG/JSON pairs
- 90,668 JPGs failed to convert (OpenCV couldn't read - corrupted downloads)
- 7,177 tensor files are 0 bytes (disk write errors during conversion)
- Solution: Add try/except in DataLoader `__getitem__` to return zeros for bad files

#### Batch Size and Learning Rate

| Batch | LR | Result |
|-------|-----|--------|
| 256 | 0.0001 | Stable, slow epochs |
| 1024 | 0.0004 | Unstable, val error spiked |
| 1024 | 0.0002 | Stable, 4× faster epochs |

Linear LR scaling (4× batch → 4× LR) was too aggressive; 2× LR worked.

#### DataLoader Memory Issues (OOM Kills)

With g5.xlarge (15GB RAM, 23GB VRAM), aggressive prefetching caused OOM:
```
workers=4, prefetch_factor=8 → 32GB virtual memory → OOM killed
workers=2, prefetch_factor=2 → 2GB memory → stable
```

Each worker prefetches `prefetch_factor` batches into shared memory. With batch_size=1024 and large tensors, this exhausts system RAM even though GPU VRAM is fine.

#### Current Training Setup (AWS)

```bash
python3 train_tensors_fixed.py \
  --checkpoint /opt/dlami/nvme/full_training/best_model.pth \
  --data /opt/dlami/nvme/wiki_tensors \
  --batch_size 1024 \
  --lr 0.0002 \
  --workers 2 \
  --prefetch 2 \
  --epochs 500
```

- Instance: g5.xlarge (A10G 23GB VRAM, 4 vCPU, 15GB RAM)
- Dataset: 861K training / 96K validation tensors
- Epoch time: ~10 minutes
- Best validation error: **24.3px** (epoch 20, 2025-12-18)

### Trained Models

Pre-trained corner detection models are included in the `models/` directory:

```
models/
├── corners_model_24.3px.onnx    # ONNX format (recommended for OpenCV DNN)
├── corners_model_24.3px.pt      # TorchScript format (for DJL)
├── full_training_24.3px_epoch20.pth  # PyTorch state dict (best)
└── training_log_24.3px.txt      # Training log
```

**Model Input/Output:**
- Input: 240×135 grayscale image, normalized to [0,1]
- Output: 8 floats (4 corner points as x,y pairs, normalized to [-1,1])
- To convert output to pixels: `x_px = (x + 1) * width / 2`, `y_px = (y + 1) * height / 2`

**Full checkpoint archive:** `/Volumes/SamsungBlue/ml-training/progressive_checkpoints/`

### Java Inference Options

The trained model can be used in Java through several approaches:

#### Model Formats

All formats can be derived from `.pth` + model code. No format *must* be exported from AWS.

| Format | File | Size | Created From |
|--------|------|------|--------------|
| PyTorch state dict | `.pth` | 4.7 MB | Training output |
| JSON weights | `.json` | 27 MB | Convert from .pth locally |
| TorchScript | `.pt` | 4.7 MB | Convert from .pth locally |
| ONNX | `.onnx` | 4.6 MB | Convert from .pth locally |

#### Conversion Scripts

```python
# .pth → TorchScript (trace-based, works without source inspection)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
traced = torch.jit.trace(model, torch.randn(1, 1, 135, 240))
traced.save("model.pt")

# .pth → ONNX
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["image"], output_names=["corners"],
                  dynamic_axes={"image": {0: "batch"}, "corners": {0: "batch"}})

# .pth → JSON (for pure Java implementation)
import json
state = torch.load("model.pth", weights_only=True)
json_state = {k: {"shape": list(v.shape), "data": v.numpy().tolist()} for k, v in state.items()}
json.dump(json_state, open("model.json", "w"))
```

#### Java Inference Approaches

**1. OpenCV DNN + ONNX (Recommended)**
- Already have OpenCV in project
- `Dnn.readNetFromONNX("model.onnx")`
- Good performance, minimal dependencies

**2. DJL (Deep Java Library)**
- Amazon's ML library for Java
- Loads TorchScript `.pt` files
- Native PyTorch execution
- Add dependency: `ai.djl.pytorch:pytorch-engine`

**3. ONNX Runtime Java**
- Microsoft's cross-platform inference
- Load ONNX models directly
- Good for production deployment
- Add dependency: `com.microsoft.onnxruntime:onnxruntime`

**4. Pure Java Implementation**
- Implement ResNet architecture manually (~200 lines)
- Load weights from JSON file
- No external ML dependencies
- Most control, useful for understanding/debugging
