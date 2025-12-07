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

    @Override
    public void start(Stage primaryStage) {
        // Parse command line arguments
        java.util.List<String> params = getParameters().getRaw();
        for (String param : params) {
            if ("--start".equals(param)) {
                autoStartPipeline = true;
            } else if (!param.startsWith("--")) {
                // Non-flag argument is the pipeline file path
                commandLinePipelineFile = param;
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
            System.out.println("Auto-start pipeline requested via --start command line or exec:exec@start in pom.xml target");
        }
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
            // Auto-start pipeline if --start was passed
            if (autoStartPipeline && !pipelineRunning) {
                startPipeline();
            }
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
        MenuItem searchHelpItem = new MenuItem("Search OpenCV Pipeline Help...");
        searchHelpItem.setOnAction(e -> FXHelpBrowser.openSearch(primaryStage));
        MenuItem aboutItem = new MenuItem("About");
        aboutItem.setOnAction(e -> FXHelpBrowser.openAbout(primaryStage));
        helpMenu.getItems().addAll(searchHelpItem, new SeparatorMenuItem(), aboutItem);

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

        // Create and start the pipeline executor
        pipelineExecutor = new FXPipelineExecutor(nodes, connections, webcamSources);
        pipelineExecutor.setBasePath(currentFilePath);  // For resolving relative paths in nested containers
        pipelineExecutor.setOnNodeOutput((node, mat) -> {
            try {
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
                pipelineExecutor.syncAllStats();

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
            webcam.stop();
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
