package com.ttennebkram.pipeline.fx;

import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.layout.BorderPane;
import javafx.stage.Modality;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.function.Supplier;

/**
 * Window for editing Container node sub-pipelines.
 * This is now a thin wrapper around FXPipelineEditor.
 */
public class FXContainerEditorWindow {

    private Stage stage;
    private FXNode containerNode;
    private Runnable onModified;
    private FXPipelineEditor editor;
    private String basePath;

    // Pipeline control callbacks (passed through to editor)
    private Runnable onStartPipeline;
    private Runnable onStopPipeline;
    private Runnable onRefreshPipeline;
    private Supplier<Boolean> isPipelineRunning;
    private Supplier<Integer> getThreadCount;
    private Runnable onRequestGlobalSave;
    private Runnable onQuit;
    private Runnable onRestart;

    /**
     * Create a new container editor window.
     *
     * @param parent The parent stage
     * @param containerNode The container node to edit
     * @param onModified Callback when content is modified
     */
    public FXContainerEditorWindow(Stage parent, FXNode containerNode, Runnable onModified) {
        this.containerNode = containerNode;
        this.onModified = onModified;

        // Ensure inner nodes and connections are initialized
        if (containerNode.innerNodes == null) {
            containerNode.innerNodes = new ArrayList<>();
        }
        if (containerNode.innerConnections == null) {
            containerNode.innerConnections = new ArrayList<>();
        }

        // Create stage
        stage = new Stage();
        stage.initOwner(parent);
        stage.initModality(Modality.NONE);
        stage.setTitle("Edit Container: " + containerNode.label);

        // Create the editor (false = not root diagram, so boundary nodes will be auto-created)
        editor = new FXPipelineEditor(
            false,  // isRootDiagram
            stage,
            containerNode.innerNodes,
            containerNode.innerConnections
        );

        // Wire up modification callback
        editor.setOnModified(() -> {
            if (this.onModified != null) {
                this.onModified.run();
            }
        });

        // Create the scene with menu bar
        BorderPane root = new BorderPane();
        root.setTop(createMenuBar());
        root.setCenter(editor.getRootPane());

        Scene scene = new Scene(root, 1200, 700);

        // Add keyboard shortcuts to scene
        scene.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.DELETE || e.getCode() == KeyCode.BACK_SPACE) {
                editor.deleteSelected();
                e.consume();
            } else if (e.getCode() == KeyCode.A && (e.isControlDown() || e.isMetaDown())) {
                editor.selectAll();
                e.consume();
            }
        });

        stage.setScene(scene);

        // Handle close
        stage.setOnCloseRequest(event -> {
            // Nothing special to do - changes are already in containerNode.innerNodes/innerConnections
        });
    }

    // ========================= CALLBACK SETTERS =========================

    public void setOnStartPipeline(Runnable onStartPipeline) {
        this.onStartPipeline = onStartPipeline;
        editor.setOnStartPipeline(onStartPipeline);
    }

    public void setOnStopPipeline(Runnable onStopPipeline) {
        this.onStopPipeline = onStopPipeline;
        editor.setOnStopPipeline(onStopPipeline);
    }

    public void setOnRefreshPipeline(Runnable onRefreshPipeline) {
        this.onRefreshPipeline = onRefreshPipeline;
        editor.setOnRefreshPipeline(onRefreshPipeline);
    }

    public void setIsPipelineRunning(Supplier<Boolean> isPipelineRunning) {
        this.isPipelineRunning = isPipelineRunning;
        editor.setIsPipelineRunning(isPipelineRunning);
    }

    public void setGetThreadCount(Supplier<Integer> getThreadCount) {
        this.getThreadCount = getThreadCount;
        editor.setGetThreadCount(getThreadCount);
    }

    public void setOnRequestGlobalSave(Runnable onRequestGlobalSave) {
        this.onRequestGlobalSave = onRequestGlobalSave;
        editor.setOnRequestGlobalSave(() -> {
            saveContainerContents();
        });
    }

    public void setOnQuit(Runnable onQuit) {
        this.onQuit = onQuit;
        editor.setOnQuit(onQuit);
    }

    public void setOnRestart(Runnable onRestart) {
        this.onRestart = onRestart;
        editor.setOnRestart(onRestart);
    }

    public void setBasePath(String basePath) {
        this.basePath = basePath;
        editor.setBasePath(basePath);
    }

    // ========================= PUBLIC METHODS =========================

    public void show() {
        stage.show();
        editor.paintCanvas();
    }

    public void updatePipelineButtonState() {
        editor.updatePipelineButtonState();
    }

    // ========================= MENU BAR =========================

    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();

        // File menu
        Menu fileMenu = new Menu("File");

        MenuItem saveItem = new MenuItem("Save");
        saveItem.setAccelerator(new KeyCodeCombination(KeyCode.S, KeyCombination.SHORTCUT_DOWN));
        saveItem.setOnAction(e -> saveContainerContents());

        MenuItem closeItem = new MenuItem("Close");
        closeItem.setAccelerator(new KeyCodeCombination(KeyCode.W, KeyCombination.SHORTCUT_DOWN));
        closeItem.setOnAction(e -> stage.close());

        fileMenu.getItems().addAll(saveItem, new SeparatorMenuItem(), closeItem);

        // Edit menu
        Menu editMenu = new Menu("Edit");

        MenuItem selectAllItem = new MenuItem("Select All");
        selectAllItem.setAccelerator(new KeyCodeCombination(KeyCode.A, KeyCombination.SHORTCUT_DOWN));
        selectAllItem.setOnAction(e -> editor.selectAll());

        editMenu.getItems().add(selectAllItem);

        menuBar.getMenus().addAll(fileMenu, editMenu);
        menuBar.setUseSystemMenuBar(true);

        return menuBar;
    }

    // ========================= SAVE LOGIC =========================

    /**
     * Resolve a pipeline file path, handling both absolute and relative paths.
     */
    private String resolvePipelinePath(String pipelinePath) {
        if (pipelinePath == null || pipelinePath.isEmpty()) {
            return pipelinePath;
        }
        java.io.File file = new java.io.File(pipelinePath);
        if (file.isAbsolute()) {
            return pipelinePath;
        }
        if (basePath != null && !basePath.isEmpty()) {
            java.io.File baseDir = new java.io.File(basePath);
            if (baseDir.isFile()) {
                baseDir = baseDir.getParentFile();
            }
            if (baseDir != null) {
                java.io.File resolved = new java.io.File(baseDir, pipelinePath);
                return resolved.getAbsolutePath();
            }
        }
        return pipelinePath;
    }

    /**
     * Save the container's contents.
     */
    private void saveContainerContents() {
        // Check if the container has an external pipeline file path
        if (containerNode.pipelineFilePath != null && !containerNode.pipelineFilePath.isEmpty()) {
            String resolvedPath = resolvePipelinePath(containerNode.pipelineFilePath);

            editor.setStatus("Saving...");

            // Create snapshot
            final java.util.List<FXNode> nodesCopy = new java.util.ArrayList<>(containerNode.innerNodes);
            final java.util.List<FXConnection> connectionsCopy = new java.util.ArrayList<>(containerNode.innerConnections);

            // Run save on background thread
            Thread saveThread = new Thread(() -> {
                try {
                    FXPipelineSerializer.save(resolvedPath, nodesCopy, connectionsCopy);

                    javafx.application.Platform.runLater(() -> {
                        editor.setStatus("Saved to: " + new java.io.File(resolvedPath).getName());
                        if (onModified != null) onModified.run();

                        // Also save parent document
                        if (onRequestGlobalSave != null) {
                            onRequestGlobalSave.run();
                        }
                    });
                } catch (Exception e) {
                    javafx.application.Platform.runLater(() -> {
                        editor.setStatus("Save failed");
                        Alert alert = new Alert(Alert.AlertType.ERROR);
                        alert.setTitle("Save Error");
                        alert.setHeaderText("Failed to save pipeline");
                        alert.setContentText("Error: " + e.getMessage() + "\n\nFile: " + resolvedPath);
                        alert.showAndWait();
                    });
                }
            }, "Container-Save-Thread");
            saveThread.setDaemon(true);
            saveThread.start();
        } else {
            // No external file - save to parent document
            if (onRequestGlobalSave != null) {
                onRequestGlobalSave.run();
                editor.setStatus("Saved to parent document");
            }
        }
    }
}
