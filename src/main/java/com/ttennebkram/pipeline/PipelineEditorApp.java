package com.ttennebkram.pipeline;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import com.ttennebkram.pipeline.fx.FXConnection;
import com.ttennebkram.pipeline.fx.FXContainerEditorWindow;
import com.ttennebkram.pipeline.fx.FXHelpBrowser;
import com.ttennebkram.pipeline.fx.FXImageUtils;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPipelineEditor;
import com.ttennebkram.pipeline.fx.FXPipelineExecutor;
import com.ttennebkram.pipeline.fx.FXPipelineSerializer;
import com.ttennebkram.pipeline.util.MatTracker;
import com.ttennebkram.pipeline.fx.FXWebcamSource;
import com.ttennebkram.pipeline.fx.NodeRenderer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.prefs.Preferences;

/**
 * Main JavaFX Application class for the Pipeline Editor.
 * Uses FXPipelineEditor for the canvas/node editing UI.
 */
public class PipelineEditorApp extends Application {

    // =========================== COLOR CONSTANTS ===========================
    private static final Color COLOR_STATUS_STOPPED = Color.rgb(180, 0, 0);
    private static final Color COLOR_STATUS_RUNNING = Color.rgb(0, 128, 0);
    // ========================================================================

    private Stage primaryStage;
    private FXPipelineEditor editor;

    // Pipeline data model - owned by this class, edited by FXPipelineEditor
    private List<FXNode> nodes = new ArrayList<>();
    private List<FXConnection> connections = new ArrayList<>();

    // Pipeline execution state
    private boolean pipelineRunning = false;
    private boolean isDirty = false;
    private String currentFilePath = null;

    // Recent files
    private static final int MAX_RECENT_FILES = 10;
    private static final String RECENT_FILES_KEY = "recentFiles";
    private static final String LAST_FILE_KEY = "lastFile";
    private Preferences prefs;
    private List<String> recentFiles = new ArrayList<>();
    private Menu openRecentMenu;

    // Webcam sources for live preview (ConcurrentHashMap for thread-safe access from executor thread)
    private Map<Integer, FXWebcamSource> webcamSources = new ConcurrentHashMap<>();

    // Pipeline executor
    private FXPipelineExecutor pipelineExecutor;

    // Command line options
    private boolean autoStartPipeline = false;
    private String commandLinePipelineFile = null;
    private String fullscreenNodeName = null;
    private long maxRuntimeSeconds = 0; // 0 = no limit

    // Global camera settings (null = use node settings)
    private Integer cmdCameraIndex = null; // -1 = auto-detect
    private Integer cmdCameraResolution = null; // 0=320x240, 1=640x480, 2=1280x720, 3=1920x1080
    private Double cmdCameraFps = null; // Any positive value, or <0 for default (1 fps)
    private Boolean cmdCameraMirror = null;

    // Auto-save behavior for --max_time exit: "prompt" (default), "yes", "no"
    private String autoSaveBehavior = "prompt";

    @Override
    public void start(Stage primaryStage) {
        // Parse command line arguments
        java.util.List<String> params = getParameters().getRaw();
        for (int i = 0; i < params.size(); i++) {
            String param = params.get(i);
            if ("-h".equals(param) || "--help".equals(param)) {
                printHelp();
                Platform.exit();
                return;
            } else if ("-a".equals(param) || "--auto_start".equals(param) || "--autostart".equals(param) || "--auto_run".equals(param) || "--autorun".equals(param) || "--start".equals(param)) {
                autoStartPipeline = true;
            } else if ("--fullscreen_node_name".equals(param)) {
                if (i + 1 < params.size()) {
                    fullscreenNodeName = params.get(++i);
                } else {
                    System.err.println("Error: --fullscreen_node_name requires a node name argument");
                    System.err.println("       Tip: You can set a node's name by double-clicking on it.");
                    Platform.exit();
                    return;
                }
            } else if ("--max_time".equals(param)) {
                if (i + 1 < params.size()) {
                    try {
                        maxRuntimeSeconds = Long.parseLong(params.get(++i));
                    } catch (NumberFormatException e) {
                        System.err.println("Error: --max_time requires a numeric value in seconds");
                        Platform.exit();
                        return;
                    }
                } else {
                    System.err.println("Error: --max_time requires a value in seconds");
                    Platform.exit();
                    return;
                }
            } else if ("--autosave_yes".equals(param)) {
                autoSaveBehavior = "yes";
            } else if ("--autosave_no".equals(param)) {
                autoSaveBehavior = "no";
            } else if ("--autosave_prompt".equals(param) || "--autosave_default".equals(param)) {
                autoSaveBehavior = "prompt";
            } else if ("--camera_index".equals(param)) {
                if (i + 1 < params.size()) {
                    try {
                        cmdCameraIndex = Integer.parseInt(params.get(++i));
                    } catch (NumberFormatException e) {
                        System.err.println("Error: --camera_index requires a numeric value (-1 for auto-detect)");
                        Platform.exit();
                        return;
                    }
                } else {
                    System.err.println("Error: --camera_index requires a value (-1 for auto-detect)");
                    Platform.exit();
                    return;
                }
            } else if ("--camera_resolution".equals(param)) {
                if (i + 1 < params.size()) {
                    String resArg = params.get(++i);
                    switch (resArg) {
                        case "320x240": case "0": cmdCameraResolution = 0; break;
                        case "640x480": case "1": cmdCameraResolution = 1; break;
                        case "1280x720": case "720p": case "2": cmdCameraResolution = 2; break;
                        case "1920x1080": case "1080p": case "3": cmdCameraResolution = 3; break;
                        default:
                            System.err.println("Error: --camera_resolution must be 320x240, 640x480, 1280x720, or 1920x1080");
                            Platform.exit();
                            return;
                    }
                } else {
                    System.err.println("Error: --camera_resolution requires a value (e.g., 640x480 or 1080p)");
                    Platform.exit();
                    return;
                }
            } else if ("--camera_fps".equals(param)) {
                if (i + 1 < params.size()) {
                    try {
                        cmdCameraFps = Double.parseDouble(params.get(++i));
                    } catch (NumberFormatException e) {
                        System.err.println("Error: --camera_fps requires a numeric value");
                        Platform.exit();
                        return;
                    }
                } else {
                    System.err.println("Error: --camera_fps requires a value");
                    Platform.exit();
                    return;
                }
            } else if ("--camera_mirror".equals(param)) {
                if (i + 1 < params.size()) {
                    String mirrorArg = params.get(++i).toLowerCase();
                    if ("true".equals(mirrorArg) || "yes".equals(mirrorArg) || "1".equals(mirrorArg)) {
                        cmdCameraMirror = true;
                    } else if ("false".equals(mirrorArg) || "no".equals(mirrorArg) || "0".equals(mirrorArg)) {
                        cmdCameraMirror = false;
                    } else {
                        System.err.println("Error: --camera_mirror must be true/false or yes/no");
                        Platform.exit();
                        return;
                    }
                } else {
                    System.err.println("Error: --camera_mirror requires a value (true/false)");
                    Platform.exit();
                    return;
                }
            } else if (!param.startsWith("-")) {
                // Non-flag argument is the pipeline file path
                commandLinePipelineFile = param;
            } else {
                System.err.println("Unknown option: " + param);
                printHelp();
                Platform.exit();
                return;
            }
        }
        this.primaryStage = primaryStage;

        // Initialize preferences
        prefs = Preferences.userNodeForPackage(PipelineEditorApp.class);
        loadRecentFiles();

        // Create the editor component (true = root diagram)
        editor = new FXPipelineEditor(true, primaryStage, nodes, connections);

        // Wire up callbacks
        editor.setOnModified(this::markDirty);
        editor.setOnStartPipeline(this::startPipeline);
        editor.setOnStopPipeline(this::stopPipeline);
        editor.setOnRefreshPipeline(this::refreshPipeline);
        editor.setIsPipelineRunning(() -> pipelineRunning);
        editor.setGetThreadCount(this::getThreadCount);
        editor.setOnRequestGlobalSave(this::saveDiagram);
        editor.setOnQuit(() -> {
            stopPipeline();
            Platform.exit();
        });
        editor.setOnRestart(this::restartApplication);

        // Create main layout
        BorderPane root = new BorderPane();

        // Menu bar at top
        root.setTop(createMenuBar());

        // Editor component fills the center
        root.setCenter(editor.getRootPane());

        // Create scene
        Scene scene = new Scene(root, 1400, 800);

        // Add keyboard handler for delete key, arrow keys, and other shortcuts
        scene.setOnKeyPressed(e -> {
            // Ctrl+M: Toggle Mat tracking
            if (e.getCode() == KeyCode.M && e.isControlDown() && !e.isShiftDown()) {
                MatTracker.setEnabled(!MatTracker.isEnabled());
                e.consume();
                return;
            }
            // Ctrl+Shift+M: Dump Mat leaks
            if (e.getCode() == KeyCode.M && e.isControlDown() && e.isShiftDown()) {
                MatTracker.printSummary();
                MatTracker.dumpLeaksByLocation();
                e.consume();
                return;
            }
            if (e.getCode() == KeyCode.DELETE || e.getCode() == KeyCode.BACK_SPACE) {
                editor.deleteSelected();
                e.consume();
            }
            // Arrow keys move selected nodes (prevents scroll pane from stealing the event)
            if (e.getCode() == KeyCode.UP || e.getCode() == KeyCode.DOWN ||
                e.getCode() == KeyCode.LEFT || e.getCode() == KeyCode.RIGHT) {
                if (editor.handleArrowKey(e.getCode())) {
                    e.consume();
                }
            }
            // F key: fullscreen preview of selected node
            if (e.getCode() == KeyCode.F && !e.isControlDown() && !e.isShiftDown() && !e.isAltDown()) {
                if (editor.showFullscreenPreview()) {
                    e.consume();
                }
            }
        });

        primaryStage.setTitle("OpenCV Pipeline Editor - (untitled)");
        primaryStage.setScene(scene);

        // Center on screen
        primaryStage.centerOnScreen();

        // Handle close request
        primaryStage.setOnCloseRequest(event -> {
            if (!checkUnsavedChanges()) {
                event.consume(); // Cancel close
            } else {
                stopPipeline();
                Platform.exit();
            }
        });

        primaryStage.show();

        // Initial canvas paint
        Platform.runLater(editor::paintCanvas);

        // Load last opened file if it exists
        Platform.runLater(this::loadLastFile);

        System.out.println("JavaFX Pipeline Editor started");
        System.out.println("OpenCV version: " + org.opencv.core.Core.VERSION);
        if (autoStartPipeline) {
            System.out.println("Auto-start pipeline requested via -a/--autostart command line");
        }
        if (fullscreenNodeName != null) {
            System.out.println("Fullscreen node requested: " + fullscreenNodeName);
        }
        if (maxRuntimeSeconds > 0) {
            System.out.println("Max runtime: " + maxRuntimeSeconds + " seconds");
            if (!autoStartPipeline && fullscreenNodeName == null) {
                System.err.println("Warning: --max_time specified without -a/--auto_start or --fullscreen_node_name.");
                System.err.println("         Application may auto-exit unexpectedly, after " + maxRuntimeSeconds + " seconds.");
                // Show dialog to user
                Platform.runLater(() -> {
                    javafx.scene.control.Alert alert = new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.WARNING);
                    alert.setTitle("Warning");
                    alert.setHeaderText("Unusual command line options");
                    alert.setContentText("--max_time specified without -a/--auto_start or --fullscreen_node_name.\n\nApplication may auto-exit unexpectedly, after " + maxRuntimeSeconds + " seconds.");
                    alert.showAndWait();
                });
            }
        }
        if (fullscreenNodeName != null && !autoStartPipeline) {
            System.err.println("Warning: --fullscreen_node_name specified without -a/--auto_start.");
            System.err.println("         Fullscreen preview requires a running pipeline to display images.");
            // Show dialog to user
            Platform.runLater(() -> {
                javafx.scene.control.Alert alert = new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.WARNING);
                alert.setTitle("Warning");
                alert.setHeaderText("Fullscreen without auto-start");
                alert.setContentText("--fullscreen_node_name specified without -a/--auto_start.\n\nFullscreen preview requires a running pipeline to display images.");
                alert.showAndWait();
            });
        }
    }

    private void printHelp() {
        System.out.println("OpenCV Pipeline Editor");
        System.out.println();
        System.out.println("Usage: java -jar opencv-pipeline.jar [options] [pipeline.json]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -h, --help                     Show this help message and exit");
        System.out.println("  -a, --auto_start               Automatically start the pipeline after loading");
        System.out.println("  --fullscreen_node_name NAME    Show fullscreen preview of node with given name");
        System.out.println("  --max_time SECONDS             Exit after specified number of seconds");
        System.out.println();
        System.out.println("Camera options (override all webcam source nodes):");
        System.out.println("  --camera_index INDEX           Camera index (-1 for auto-detect)");
        System.out.println("  --camera_resolution RES        Resolution: 320x240, 640x480, 1280x720, 1920x1080");
        System.out.println("  --camera_fps FPS               Target frame rate (any number, -1 for default 1fps)");
        System.out.println("  --camera_mirror BOOL           Mirror horizontally: true/false");
        System.out.println();
        System.out.println("Save behavior options (for use with --max_time):");
        System.out.println("  --autosave_prompt              Show save dialog if unsaved changes (default)");
        System.out.println("  --autosave_yes                 Automatically save without prompting");
        System.out.println("  --autosave_no                  Exit without saving");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  java -jar opencv-pipeline.jar");
        System.out.println("  java -jar opencv-pipeline.jar pipeline.json");
        System.out.println("  java -jar opencv-pipeline.jar pipeline.json -a");
        System.out.println("  java -jar opencv-pipeline.jar pipeline.json -a --camera_index 2 --camera_resolution 1080p");
        System.out.println("  java -jar opencv-pipeline.jar pipeline.json -a --fullscreen_node_name \"Monitor\" --max_time 60");
    }

    private void loadLastFile() {
        // Command line file takes precedence over last file
        String fileToLoad = null;
        if (commandLinePipelineFile != null) {
            File cmdFile = new File(commandLinePipelineFile);
            if (cmdFile.exists()) {
                fileToLoad = cmdFile.getAbsolutePath();
            } else {
                System.err.println("Pipeline file not found: " + commandLinePipelineFile);
            }
        }

        // Fall back to last file if no command line file
        if (fileToLoad == null) {
            String lastFile = prefs.get(LAST_FILE_KEY, null);
            if (lastFile != null && new File(lastFile).exists()) {
                fileToLoad = lastFile;
            }
        }

        if (fileToLoad != null) {
            loadDiagramFromPath(fileToLoad);
            // Auto-start pipeline if -a/--autostart was passed
            if (autoStartPipeline && !pipelineRunning) {
                startPipeline();

                // If fullscreen node requested, wait a moment for images to generate then show fullscreen
                if (fullscreenNodeName != null) {
                    // Delay to allow pipeline to produce first frame
                    new Thread(() -> {
                        try {
                            Thread.sleep(1000); // Wait 1 second for first frame
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        Platform.runLater(this::showFullscreenForNamedNode);
                    }).start();
                }

                // If max_time specified, start exit timer
                if (maxRuntimeSeconds > 0) {
                    startExitTimer();
                }
            }
        }
    }

    /**
     * Find node by label and show fullscreen preview.
     */
    private void showFullscreenForNamedNode() {
        if (fullscreenNodeName == null) return;

        // Find all nodes matching the name
        java.util.List<FXNode> matchingNodes = new java.util.ArrayList<>();
        for (FXNode node : nodes) {
            if (fullscreenNodeName.equals(node.label)) {
                matchingNodes.add(node);
            }
        }

        if (matchingNodes.isEmpty()) {
            System.err.println("Warning: Node not found for fullscreen: " + fullscreenNodeName);
            // Show dialog to user
            javafx.scene.control.Alert alert = new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.WARNING);
            alert.setTitle("Node Not Found");
            alert.setHeaderText("Fullscreen node not found");
            alert.setContentText("No node with name \"" + fullscreenNodeName + "\" was found in the pipeline.\n\nThe pipeline will continue running without fullscreen preview.\n\nTip: You can set a node's name by double-clicking on it.");
            alert.showAndWait();
            return;
        }

        if (matchingNodes.size() > 1) {
            System.err.println("Warning: Multiple nodes found with name \"" + fullscreenNodeName + "\" (" + matchingNodes.size() + " matches).");
            System.err.println("         Choosing node at position (" + matchingNodes.get(0).x + ", " + matchingNodes.get(0).y + ") (pseudo-random)");
        }

        FXNode chosen = matchingNodes.get(0);
        editor.selectNode(chosen);
        editor.showFullscreenPreview();
    }

    /**
     * Start timer to exit after maxRuntimeSeconds.
     */
    private void startExitTimer() {
        Thread timerThread = new Thread(() -> {
            try {
                Thread.sleep(maxRuntimeSeconds * 1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            System.out.println("Max runtime reached (" + maxRuntimeSeconds + "s), exiting...");
            Platform.runLater(this::handleMaxTimeExit);
        });
        timerThread.setDaemon(true);  // Don't prevent JVM exit
        timerThread.start();
    }

    /**
     * Handle exit when max_time is reached, respecting autosave behavior.
     */
    private void handleMaxTimeExit() {
        stopPipeline();

        if ("yes".equals(autoSaveBehavior)) {
            // Auto-save without prompting
            if (isDirty && currentFilePath != null) {
                saveDiagramToPath(currentFilePath);
                System.out.println("Auto-saved to: " + currentFilePath);
            }
            Platform.exit();
        } else if ("no".equals(autoSaveBehavior)) {
            // Exit without saving
            Platform.exit();
        } else {
            // "prompt" - show the normal unsaved changes dialog
            if (checkUnsavedChanges()) {
                Platform.exit();
            }
            // If user cancels, don't exit (but timer has expired, so just exit anyway)
            // Actually for max_time, we should still exit even if they cancel
            Platform.exit();
        }
    }

    // ========================= MENU BAR =========================

    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();

        // File menu
        Menu fileMenu = new Menu("File");

        MenuItem newItem = new MenuItem("New");
        newItem.setAccelerator(new KeyCodeCombination(KeyCode.N, KeyCombination.SHORTCUT_DOWN));
        newItem.setOnAction(e -> newDiagram());

        MenuItem openItem = new MenuItem("Open...");
        openItem.setAccelerator(new KeyCodeCombination(KeyCode.O, KeyCombination.SHORTCUT_DOWN));
        openItem.setOnAction(e -> loadDiagram());

        openRecentMenu = new Menu("Open Recent");
        updateOpenRecentMenu();

        MenuItem saveItem = new MenuItem("Save");
        saveItem.setAccelerator(new KeyCodeCombination(KeyCode.S, KeyCombination.SHORTCUT_DOWN));
        saveItem.setOnAction(e -> saveDiagram());

        MenuItem saveAsItem = new MenuItem("Save As...");
        saveAsItem.setAccelerator(new KeyCodeCombination(KeyCode.S, KeyCombination.SHORTCUT_DOWN, KeyCombination.SHIFT_DOWN));
        saveAsItem.setOnAction(e -> saveDiagramAs());

        MenuItem restartItem = new MenuItem("Restart");
        restartItem.setOnAction(e -> restartApplication());

        MenuItem quitItem = new MenuItem("Quit");
        quitItem.setAccelerator(new KeyCodeCombination(KeyCode.Q, KeyCombination.SHORTCUT_DOWN));
        quitItem.setOnAction(e -> {
            if (!checkUnsavedChanges()) {
                return;
            }
            stopPipeline();
            Platform.exit();
        });

        fileMenu.getItems().addAll(newItem, openItem, openRecentMenu, new SeparatorMenuItem(), saveItem, saveAsItem,
                new SeparatorMenuItem(), quitItem, restartItem);

        // Edit menu
        Menu editMenu = new Menu("Edit");

        MenuItem deleteItem = new MenuItem("Delete");
        deleteItem.setAccelerator(new KeyCodeCombination(KeyCode.BACK_SPACE));
        deleteItem.setOnAction(e -> editor.deleteSelected());

        MenuItem selectAllItem = new MenuItem("Select All");
        selectAllItem.setAccelerator(new KeyCodeCombination(KeyCode.A, KeyCombination.SHORTCUT_DOWN));
        selectAllItem.setOnAction(e -> editor.selectAll());

        editMenu.getItems().addAll(deleteItem, selectAllItem);

        // Help menu
        Menu helpMenu = new Menu("Help");
        MenuItem readmeItem = new MenuItem("README");
        readmeItem.setOnAction(e -> FXHelpBrowser.openReadme(primaryStage));
        MenuItem searchHelpItem = new MenuItem("Search OpenCV Pipeline Help...");
        searchHelpItem.setOnAction(e -> FXHelpBrowser.openSearch(primaryStage));
        MenuItem aboutItem = new MenuItem("About");
        aboutItem.setOnAction(e -> FXHelpBrowser.openAbout(primaryStage));
        helpMenu.getItems().addAll(readmeItem, searchHelpItem, new SeparatorMenuItem(), aboutItem);

        menuBar.getMenus().addAll(fileMenu, editMenu, helpMenu);

        // macOS-specific menu bar integration
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("mac")) {
            menuBar.setUseSystemMenuBar(true);
        }

        return menuBar;
    }

    // ========================= FILE OPERATIONS =========================

    private void newDiagram() {
        if (!checkUnsavedChanges()) {
            return;
        }

        // Stop pipeline first
        stopPipeline();
        stopAllWebcams();

        // Close all open container editor windows
        editor.closeAllContainerWindows();

        nodes.clear();
        connections.clear();
        currentFilePath = null;
        isDirty = false;
        updateTitle();
        editor.paintCanvas();
        editor.updateStatus();
    }

    private void loadDiagram() {
        if (!checkUnsavedChanges()) {
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Pipeline");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Pipeline Files", "*.json"));
        File file = fileChooser.showOpenDialog(primaryStage);

        if (file != null) {
            loadDiagramFromPath(file.getAbsolutePath());
        }
    }

    private void loadDiagramFromPath(String path) {
        // Stop pipeline first
        stopPipeline();
        stopAllWebcams();

        // Close all open container editor windows
        editor.closeAllContainerWindows();

        try {
            FXPipelineSerializer.PipelineDocument doc = FXPipelineSerializer.load(path);
            nodes.clear();
            connections.clear();
            nodes.addAll(doc.nodes);
            connections.addAll(doc.connections);

            // Mark all root-level nodes as not embedded (they're at the top level, not inside a container)
            for (FXNode node : nodes) {
                node.isEmbedded = false;
            }

            currentFilePath = path;
            editor.setBasePath(path);
            isDirty = false;

            // Add to recent files
            addRecentFile(path);
            prefs.put(LAST_FILE_KEY, path);

            updateTitle();
            editor.paintCanvas();
            editor.updateStatus();

            System.out.println("Loaded pipeline: " + path + " (" + nodes.size() + " nodes, " + connections.size() + " connections)");
        } catch (Exception e) {
            checkAndReportTooManyFiles(e);
            e.printStackTrace();
            showError("Load Error", "Failed to load pipeline: " + e.getMessage());
        }
    }

    private void saveDiagram() {
        if (currentFilePath == null) {
            saveDiagramAs();
        } else {
            saveDiagramToPath(currentFilePath);
        }
    }

    private void saveDiagramAs() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Pipeline");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Pipeline Files", "*.json"));
        if (currentFilePath != null) {
            fileChooser.setInitialDirectory(new File(currentFilePath).getParentFile());
            fileChooser.setInitialFileName(new File(currentFilePath).getName());
        }
        File file = fileChooser.showSaveDialog(primaryStage);

        if (file != null) {
            String path = file.getAbsolutePath();
            if (!path.endsWith(".json")) {
                path += ".json";
            }
            saveDiagramToPath(path);
        }
    }

    private void saveDiagramToPath(String path) {
        editor.setStatus("Saving...");

        // Create snapshot of current state
        final List<FXNode> nodesCopy = new ArrayList<>(nodes);
        final List<FXConnection> connectionsCopy = new ArrayList<>(connections);

        // Run save on background thread to avoid blocking UI
        Thread saveThread = new Thread(() -> {
            try {
                FXPipelineSerializer.save(path, nodesCopy, connectionsCopy);

                Platform.runLater(() -> {
                    currentFilePath = path;
                    editor.setBasePath(path);
                    isDirty = false;
                    addRecentFile(path);
                    prefs.put(LAST_FILE_KEY, path);
                    updateTitle();
                    editor.setStatus("Saved to: " + new File(path).getName());
                    System.out.println("Saved pipeline: " + path);
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    editor.setStatus("Save failed");
                    checkAndReportTooManyFiles(e);
                    e.printStackTrace();
                    showError("Save Error", "Failed to save pipeline: " + e.getMessage());
                });
            }
        }, "PipelineSaveThread");
        saveThread.setDaemon(true);
        saveThread.start();
    }

    private void checkAndReportTooManyFiles(Throwable t) {
        String msg = t.getMessage();
        if (msg != null && (msg.contains("Too many open files") || msg.contains("EMFILE"))) {
            System.err.println("\n*** TOO MANY OPEN FILES ERROR ***");
            System.err.println("This usually means file handles are being leaked.");
            System.err.println("Check for unclosed InputStreams, especially in image loading code.");
            System.err.println("Current open file limit: " + getOpenFileLimit());
            System.err.println("***\n");
        }
    }

    private String getOpenFileLimit() {
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"sh", "-c", "ulimit -n"});
            java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(p.getInputStream()));
            String line = reader.readLine();
            reader.close();
            return line;
        } catch (Exception e) {
            return "unknown";
        }
    }

    // ========================= PIPELINE EXECUTION =========================

    private void togglePipeline() {
        if (pipelineRunning) {
            stopPipeline();
        } else {
            startPipeline();
        }
    }

    private void startPipeline() {
        pipelineRunning = true;
        editor.updatePipelineButtonState();

        // Clear all node and connection stats before starting
        clearPipelineStats();

        // Create webcam sources for all webcam nodes
        for (FXNode node : nodes) {
            if ("WebcamSource".equals(node.nodeType)) {
                // Get camera settings from node properties, with command line overrides
                int cameraIndex = -1;
                int resolutionIndex = 1;
                int fpsIndex = 0;
                boolean mirror = true;

                // Command line options override node settings
                if (cmdCameraIndex != null) {
                    cameraIndex = cmdCameraIndex;
                } else {
                    Object camIdx = node.properties.get("cameraIndex");
                    if (camIdx instanceof Number) cameraIndex = ((Number) camIdx).intValue();
                }

                if (cmdCameraResolution != null) {
                    resolutionIndex = cmdCameraResolution;
                } else {
                    Object resIdx = node.properties.get("resolutionIndex");
                    if (resIdx instanceof Number) resolutionIndex = ((Number) resIdx).intValue();
                }

                // FPS: command line can be direct value or use node's index
                Double directFps = null;
                if (cmdCameraFps != null) {
                    if (cmdCameraFps < 0) {
                        fpsIndex = 0; // Default to 1 fps
                    } else {
                        directFps = cmdCameraFps; // Will use setFps() later
                    }
                } else {
                    Object fps = node.properties.get("fpsIndex");
                    if (fps instanceof Number) fpsIndex = ((Number) fps).intValue();
                }

                if (cmdCameraMirror != null) {
                    mirror = cmdCameraMirror;
                } else {
                    Object mirrorVal = node.properties.get("mirrorHorizontal");
                    if (mirrorVal instanceof Boolean) mirror = (Boolean) mirrorVal;
                }

                FXWebcamSource webcam;

                // On Raspberry Pi with libcamera/V4L2, camera indices can be unstable
                // and reopening a recently-closed device can fail due to timing issues.
                // If a specific index is requested (>= 0), try it first.
                // If index is -1, use auto-detect which finds first camera returning non-blank frames.
                if (cameraIndex >= 0) {
                    // Try the specific camera first
                    webcam = FXWebcamSource.findAndOpenCamera(cameraIndex);
                } else {
                    // Auto-detect: find first camera that returns non-blank frames
                    webcam = FXWebcamSource.findAndOpenCamera();
                }

                if (webcam != null) {
                    cameraIndex = webcam.getCameraIndex();
                    // Store actual camera index in node for display
                    node.cameraIndex = cameraIndex;
                }

                if (webcam != null) {
                    // Configure the webcam source
                    webcam.setResolutionIndex(resolutionIndex);
                    if (directFps != null) {
                        webcam.setFps(directFps);
                    } else {
                        webcam.setFpsIndex(fpsIndex);
                    }
                    webcam.setMirrorHorizontal(mirror);

                    // Start the webcam
                    webcam.start();
                    webcamSources.put(node.id, webcam);
                    System.out.println("Started webcam for node " + node.label + " (camera " + cameraIndex + ")");
                } else {
                    System.err.println("Failed to open webcam for node " + node.label + " (camera " + cameraIndex + ")");
                }
            }
        }

        // Create and start the pipeline executor
        pipelineExecutor = new FXPipelineExecutor(nodes, connections, webcamSources);
        pipelineExecutor.setBasePath(currentFilePath);  // For resolving relative paths in nested containers
        pipelineExecutor.setOnNodeOutput((node, mat) -> {
            try {
                // Check if pipeline was stopped while callback was queued
                if (pipelineExecutor == null) {
                    return;
                }

                // Update node thumbnail (small, for node box display)
                javafx.scene.image.Image newThumb = FXImageUtils.matToImage(mat,
                    NodeRenderer.PROCESSING_NODE_THUMB_WIDTH,
                    NodeRenderer.PROCESSING_NODE_THUMB_HEIGHT);
                node.thumbnail = newThumb;

                // Update preview image (full resolution, cached for later display)
                javafx.scene.image.Image fullRes = FXImageUtils.matToImage(mat);
                node.previewImage = fullRes;

                // Update preview pane if this node is selected
                if (editor.getSelectedNodes().contains(node) && editor.getSelectedNodes().size() == 1) {
                    editor.getPreviewImageView().setImage(fullRes);
                }

                // Sync queue and processor stats before repainting
                if (pipelineExecutor != null) {
                    pipelineExecutor.syncAllStats();
                }

                // Repaint canvas
                editor.paintCanvas();
            } finally {
                // Always release the mat
                mat.release();
            }
        });
        pipelineExecutor.start();

        // Update status after executor has started and processors are created
        updatePipelineStatus();
    }

    private void refreshPipeline() {
        // Currently a no-op - executor doesn't have refresh capability yet
        // This callback is used by nested editors to notify parent of property changes
    }

    public static void main(String[] args) {
        launch(args);
    }

    private int getThreadCount() {
        // Count threads: processor threads + 1 for JavaFX thread + 1 for each active webcam
        return (pipelineExecutor != null ? pipelineExecutor.getThreadCount() : 0) + 1 + webcamSources.size();
    }

    private void updatePipelineStatus() {
        int threadCount = getThreadCount();
        editor.setStatus("Pipeline running (" + threadCount + " thread" + (threadCount != 1 ? "s" : "") + ")");
    }

    private void stopPipeline() {
        pipelineRunning = false;
        editor.updatePipelineButtonState();
        editor.setStatus("Stopping pipeline...");

        // Stop the pipeline executor in a background thread to avoid blocking the UI
        // The stop() method contains blocking join() calls that can cause the beach ball
        final FXPipelineExecutor executorToStop = pipelineExecutor;
        pipelineExecutor = null;

        // Close all webcams to release the VideoCapture resources
        // This must be done so cameras can be reopened on next start
        stopAllWebcams();

        if (executorToStop != null) {
            new Thread(() -> {
                executorToStop.stop();
                // Update status on the JavaFX thread when done
                Platform.runLater(() -> {
                    editor.setStatus("Pipeline stopped");
                });
            }, "PipelineStopThread").start();
        } else {
            editor.setStatus("Pipeline stopped");
        }
    }

    /**
     * Clear all pipeline statistics (node counters, connection queue stats).
     * Called when pipeline starts to reset counters from previous runs.
     */
    private void clearPipelineStats() {
        // Clear node counters
        for (FXNode node : nodes) {
            node.inputCount = 0;
            node.outputCount1 = 0;
            node.outputCount2 = 0;
            node.outputCount3 = 0;
            node.outputCount4 = 0;
            node.statusText = "";
        }
        // Clear connection queue stats
        for (FXConnection conn : connections) {
            conn.queueSize = 0;
            conn.totalFrames = 0;
        }
    }

    // ========================= WEBCAM MANAGEMENT =========================

    private void stopAllWebcams() {
        for (FXWebcamSource webcam : webcamSources.values()) {
            webcam.close();  // close() calls stop() and releases the VideoCapture
        }
        webcamSources.clear();
    }

    // ========================= APPLICATION LIFECYCLE =========================

    private void restartApplication() {
        // Check for unsaved changes first
        if (!checkUnsavedChanges()) {
            return;
        }

        // Stop pipeline first
        stopPipeline();

        try {
            // Get the java command from current JVM
            String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";

            // Get the classpath
            String classpath = System.getProperty("java.class.path");

            // Get the main class - use the launcher class to properly bootstrap JavaFX
            String mainClass = "com.ttennebkram.pipeline.PipelineEditorLauncher";

            // Build the command
            ProcessBuilder builder = new ProcessBuilder(
                javaBin,
                "-cp", classpath,
                mainClass
            );

            // Start new process
            builder.start();

            // Exit current process
            Platform.exit();
            System.exit(0);

        } catch (Exception e) {
            e.printStackTrace();
            showError("Restart Error", "Failed to restart application: " + e.getMessage());
        }
    }

    // ========================= STATE MANAGEMENT =========================

    private void markDirty() {
        isDirty = true;
        updateTitle();
    }

    private void updateTitle() {
        String title = "OpenCV Pipeline Editor - ";
        if (currentFilePath != null) {
            title += new File(currentFilePath).getName();
        } else {
            title += "(untitled)";
        }
        if (isDirty) {
            title += " *";
        }
        primaryStage.setTitle(title);
    }

    private boolean checkUnsavedChanges() {
        if (!isDirty) {
            return true;
        }

        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle("Unsaved Changes");
        alert.setHeaderText("You have unsaved changes.");
        alert.setContentText("Do you want to save before continuing?");

        ButtonType saveButton = new ButtonType("Save");
        ButtonType dontSaveButton = new ButtonType("Don't Save");
        ButtonType cancelButton = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);

        alert.getButtonTypes().setAll(saveButton, dontSaveButton, cancelButton);

        java.util.Optional<ButtonType> result = alert.showAndWait();
        if (result.isPresent()) {
            if (result.get() == saveButton) {
                saveDiagram();
                return true;
            } else if (result.get() == dontSaveButton) {
                return true;
            }
        }
        return false;
    }

    // ========================= RECENT FILES =========================

    private void loadRecentFiles() {
        String files = prefs.get(RECENT_FILES_KEY, "");
        recentFiles.clear();
        if (!files.isEmpty()) {
            for (String file : files.split("\n")) {
                if (!file.isEmpty() && new File(file).exists()) {
                    recentFiles.add(file);
                }
            }
        }
    }

    private void saveRecentFiles() {
        StringBuilder sb = new StringBuilder();
        for (String file : recentFiles) {
            if (sb.length() > 0) sb.append("\n");
            sb.append(file);
        }
        prefs.put(RECENT_FILES_KEY, sb.toString());
    }

    private void addRecentFile(String path) {
        recentFiles.remove(path);
        recentFiles.add(0, path);
        while (recentFiles.size() > MAX_RECENT_FILES) {
            recentFiles.remove(recentFiles.size() - 1);
        }
        saveRecentFiles();
        updateOpenRecentMenu();
    }

    private void updateOpenRecentMenu() {
        openRecentMenu.getItems().clear();
        for (String path : recentFiles) {
            MenuItem item = new MenuItem(new File(path).getName());
            item.setOnAction(e -> {
                if (checkUnsavedChanges()) {
                    loadDiagramFromPath(path);
                }
            });
            openRecentMenu.getItems().add(item);
        }

        if (!recentFiles.isEmpty()) {
            openRecentMenu.getItems().add(new SeparatorMenuItem());
            MenuItem clearItem = new MenuItem("Clear Recent");
            clearItem.setOnAction(e -> {
                recentFiles.clear();
                saveRecentFiles();
                updateOpenRecentMenu();
            });
            openRecentMenu.getItems().add(clearItem);
        }

        openRecentMenu.setDisable(recentFiles.isEmpty());
    }

    // ========================= DIALOGS =========================

    private void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
