package com.example.pipeline;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.prefs.Preferences;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

public class PipelineEditor {

    private Shell shell;
    private Display display;
    private Canvas canvas;
    private Canvas previewCanvas;
    private Image previewImage;

    private List<PipelineNode> nodes = new ArrayList<>();
    private List<Connection> connections = new ArrayList<>();

    private PipelineNode selectedNode = null;
    private Point dragOffset = null;
    private boolean isDragging = false;

    // Connection drawing state
    private PipelineNode connectionSource = null;
    private Point connectionEndPoint = null;

    // Recent files
    private static final int MAX_RECENT_FILES = 10;
    private static final String RECENT_FILES_KEY = "recentFiles";
    private Preferences prefs;
    private List<String> recentFiles = new ArrayList<>();
    private Combo recentFilesCombo;
    private Menu openRecentMenu;

    public static void main(String[] args) {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();

        PipelineEditor editor = new PipelineEditor();
        editor.run();
    }

    private String currentFilePath = null;

    public void run() {
        display = new Display();
        shell = new Shell(display);
        shell.setText("OpenCV Pipeline Editor");
        shell.setSize(1400, 800);
        shell.setLayout(new GridLayout(3, false));

        // Initialize preferences and load recent files
        prefs = Preferences.userNodeForPackage(PipelineEditor.class);
        loadRecentFiles();

        // Create menu bar
        createMenuBar();

        // Setup system menu (macOS application menu)
        setupSystemMenu();

        // Left side - toolbar/palette
        createToolbar();

        // Center - canvas
        createCanvas();

        // Right side - preview panel
        createPreviewPanel();

        // Create initial sample nodes
        createSamplePipeline();

        shell.open();
        while (!shell.isDisposed()) {
            if (!display.readAndDispatch()) {
                display.sleep();
            }
        }
        display.dispose();
    }

    private void createMenuBar() {
        Menu menuBar = new Menu(shell, SWT.BAR);
        shell.setMenuBar(menuBar);

        // File menu
        MenuItem fileMenuItem = new MenuItem(menuBar, SWT.CASCADE);
        fileMenuItem.setText("File");

        Menu fileMenu = new Menu(shell, SWT.DROP_DOWN);
        fileMenuItem.setMenu(fileMenu);

        // New
        MenuItem newItem = new MenuItem(fileMenu, SWT.PUSH);
        newItem.setText("New\tCmd+N");
        newItem.setAccelerator(SWT.MOD1 + 'N');
        newItem.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                newDiagram();
            }
        });

        // Open
        MenuItem openItem = new MenuItem(fileMenu, SWT.PUSH);
        openItem.setText("Open...\tCmd+O");
        openItem.setAccelerator(SWT.MOD1 + 'O');
        openItem.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                loadDiagram();
            }
        });

        // Open Recent submenu
        MenuItem openRecentItem = new MenuItem(fileMenu, SWT.CASCADE);
        openRecentItem.setText("Open Recent");
        openRecentMenu = new Menu(shell, SWT.DROP_DOWN);
        openRecentItem.setMenu(openRecentMenu);
        updateOpenRecentMenu();

        new MenuItem(fileMenu, SWT.SEPARATOR);

        // Save
        MenuItem saveItem = new MenuItem(fileMenu, SWT.PUSH);
        saveItem.setText("Save\tCmd+S");
        saveItem.setAccelerator(SWT.MOD1 + 'S');
        saveItem.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                if (currentFilePath != null) {
                    saveDiagramToPath(currentFilePath);
                } else {
                    saveDiagramAs();
                }
            }
        });

        // Save As
        MenuItem saveAsItem = new MenuItem(fileMenu, SWT.PUSH);
        saveAsItem.setText("Save As...\tCmd+Shift+S");
        saveAsItem.setAccelerator(SWT.MOD1 + SWT.SHIFT + 'S');
        saveAsItem.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                saveDiagramAs();
            }
        });

        new MenuItem(fileMenu, SWT.SEPARATOR);

        // Run Pipeline
        MenuItem runItem = new MenuItem(fileMenu, SWT.PUSH);
        runItem.setText("Run Pipeline\tCmd+E");
        runItem.setAccelerator(SWT.MOD1 + 'E');
        runItem.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                executePipeline();
            }
        });

    }

    private void setupSystemMenu() {
        // On macOS, add Restart to the application menu (after Quit)
        Menu systemMenu = display.getSystemMenu();
        if (systemMenu != null) {
            // Add separator after Quit
            new MenuItem(systemMenu, SWT.SEPARATOR);

            // Add Restart item at the end
            MenuItem restartItem = new MenuItem(systemMenu, SWT.PUSH);
            restartItem.setText("Restart\tCmd+R");
            restartItem.setAccelerator(SWT.MOD1 + 'R');
            restartItem.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    restartApplication();
                }
            });
        }
    }

    private void restartApplication() {
        // Get the command used to launch this application
        try {
            String javaHome = System.getProperty("java.home");
            String javaBin = javaHome + File.separator + "bin" + File.separator + "java";
            String classpath = System.getProperty("java.class.path");
            String className = PipelineEditor.class.getName();

            ProcessBuilder builder = new ProcessBuilder(
                javaBin, "-XstartOnFirstThread", "-cp", classpath, className
            );
            builder.inheritIO();
            builder.start();

            // Exit current instance
            display.dispose();
            System.exit(0);
        } catch (Exception ex) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_ERROR | SWT.OK);
            mb.setText("Restart Error");
            mb.setMessage("Failed to restart: " + ex.getMessage());
            mb.open();
        }
    }

    private void loadRecentFiles() {
        recentFiles.clear();
        String saved = prefs.get(RECENT_FILES_KEY, "");
        if (!saved.isEmpty()) {
            String[] paths = saved.split("\n");
            for (String path : paths) {
                if (!path.isEmpty() && new File(path).exists()) {
                    recentFiles.add(path);
                }
            }
        }
    }

    private void saveRecentFiles() {
        StringBuilder sb = new StringBuilder();
        for (String path : recentFiles) {
            if (sb.length() > 0) sb.append("\n");
            sb.append(path);
        }
        prefs.put(RECENT_FILES_KEY, sb.toString());
    }

    private void addToRecentFiles(String path) {
        // Remove if already exists
        recentFiles.remove(path);
        // Add to front
        recentFiles.add(0, path);
        // Trim to max
        while (recentFiles.size() > MAX_RECENT_FILES) {
            recentFiles.remove(recentFiles.size() - 1);
        }
        saveRecentFiles();
        updateRecentFilesCombo(recentFilesCombo);
        updateOpenRecentMenu();
    }

    private void updateOpenRecentMenu() {
        if (openRecentMenu == null || openRecentMenu.isDisposed()) return;

        // Clear existing items
        for (MenuItem item : openRecentMenu.getItems()) {
            item.dispose();
        }

        if (recentFiles.isEmpty()) {
            MenuItem emptyItem = new MenuItem(openRecentMenu, SWT.PUSH);
            emptyItem.setText("(No recent files)");
            emptyItem.setEnabled(false);
        } else {
            for (String path : recentFiles) {
                MenuItem item = new MenuItem(openRecentMenu, SWT.PUSH);
                item.setText(new File(path).getName());
                item.setData(path);
                item.addSelectionListener(new SelectionAdapter() {
                    @Override
                    public void widgetSelected(SelectionEvent e) {
                        String filePath = (String) ((MenuItem) e.widget).getData();
                        loadDiagramFromPath(filePath);
                    }
                });
            }

            // Add separator and Clear Recent
            new MenuItem(openRecentMenu, SWT.SEPARATOR);
            MenuItem clearItem = new MenuItem(openRecentMenu, SWT.PUSH);
            clearItem.setText("Clear Recent");
            clearItem.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    recentFiles.clear();
                    saveRecentFiles();
                    updateRecentFilesCombo(recentFilesCombo);
                    updateOpenRecentMenu();
                }
            });
        }
    }

    private void updateRecentFilesCombo(Combo combo) {
        if (combo == null || combo.isDisposed()) return;
        combo.removeAll();
        if (recentFiles.isEmpty()) {
            combo.add("(No recent files)");
            combo.select(0);
            combo.setEnabled(false);
        } else {
            for (String path : recentFiles) {
                combo.add(new File(path).getName());
            }
            combo.setEnabled(true);
        }
    }

    private void loadDiagramFromPath(String path) {
        try {
            // Clear existing
            for (PipelineNode node : nodes) {
                if (node instanceof ImageSourceNode) {
                    ((ImageSourceNode) node).overlayComposite.dispose();
                }
            }
            nodes.clear();
            connections.clear();

            // Load JSON
            Gson gson = new Gson();
            JsonObject root;
            try (FileReader reader = new FileReader(path)) {
                root = gson.fromJson(reader, JsonObject.class);
            }

            // Load nodes
            JsonArray nodesArray = root.getAsJsonArray("nodes");
            for (JsonElement elem : nodesArray) {
                JsonObject nodeObj = elem.getAsJsonObject();
                int x = nodeObj.get("x").getAsInt();
                int y = nodeObj.get("y").getAsInt();
                String type = nodeObj.get("type").getAsString();

                if ("ImageSource".equals(type)) {
                    ImageSourceNode node = new ImageSourceNode(shell, display, canvas, x, y);
                    if (nodeObj.has("imagePath")) {
                        String imgPath = nodeObj.get("imagePath").getAsString();
                        node.imagePath = imgPath;
                        node.loadImage(imgPath);
                    }
                    nodes.add(node);
                } else if ("Processing".equals(type)) {
                    String name = nodeObj.get("name").getAsString();
                    ProcessingNode node = new ProcessingNode(display, name, x, y);
                    nodes.add(node);
                }
            }

            // Load connections
            JsonArray connsArray = root.getAsJsonArray("connections");
            for (JsonElement elem : connsArray) {
                JsonObject connObj = elem.getAsJsonObject();
                int sourceId = connObj.get("sourceId").getAsInt();
                int targetId = connObj.get("targetId").getAsInt();
                if (sourceId >= 0 && sourceId < nodes.size() &&
                    targetId >= 0 && targetId < nodes.size()) {
                    connections.add(new Connection(nodes.get(sourceId), nodes.get(targetId)));
                }
            }

            currentFilePath = path;
            addToRecentFiles(path);
            canvas.redraw();
            shell.setText("OpenCV Pipeline Editor - " + new File(path).getName());

        } catch (Exception e) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_ERROR | SWT.OK);
            mb.setText("Load Error");
            mb.setMessage("Failed to load: " + e.getMessage());
            mb.open();
        }
    }

    private void newDiagram() {
        // Clear existing
        for (PipelineNode node : nodes) {
            if (node instanceof ImageSourceNode) {
                ((ImageSourceNode) node).overlayComposite.dispose();
            }
        }
        nodes.clear();
        connections.clear();
        currentFilePath = null;
        shell.setText("OpenCV Pipeline Editor");
        canvas.redraw();
    }

    private void createToolbar() {
        Composite toolbar = new Composite(shell, SWT.BORDER);
        toolbar.setLayoutData(new GridData(SWT.FILL, SWT.FILL, false, true));
        toolbar.setLayout(new GridLayout(1, false));

        Font boldFont = new Font(display, "Arial", 11, SWT.BOLD);

        // Inputs
        Label inputsLabel = new Label(toolbar, SWT.NONE);
        inputsLabel.setText("Inputs:");
        inputsLabel.setFont(boldFont);

        createNodeButton(toolbar, "Image Source", () -> addImageSourceNode());

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Effects
        Label effectsLabel = new Label(toolbar, SWT.NONE);
        effectsLabel.setText("Effects:");
        effectsLabel.setFont(boldFont);

        createNodeButton(toolbar, "Gaussian Blur", () -> addProcessingNode("Gaussian Blur"));
        createNodeButton(toolbar, "Threshold", () -> addProcessingNode("Threshold"));
        createNodeButton(toolbar, "Canny Edge", () -> addProcessingNode("Canny Edge"));
        createNodeButton(toolbar, "Grayscale", () -> addProcessingNode("Grayscale"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Outputs
        Label outputsLabel = new Label(toolbar, SWT.NONE);
        outputsLabel.setText("Outputs:");
        outputsLabel.setFont(boldFont);

        createNodeButton(toolbar, "Output", () -> addProcessingNode("Output"));

        // Separator
        Label sep = new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL);
        sep.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));


        // Instructions
        Label instructions = new Label(toolbar, SWT.WRAP);
        instructions.setText("Instructions:\n\n" +
            "• Drag nodes to move\n" +
            "• Right-click to connect\n" +
            "• Click 'Choose...' for image");
        GridData gd = new GridData(SWT.FILL, SWT.FILL, true, false);
        gd.widthHint = 150;
        instructions.setLayoutData(gd);

        // Spacer to push remaining items to bottom
        Label spacer = new Label(toolbar, SWT.NONE);
        spacer.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));

        // Recent Files section
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Label recentLabel = new Label(toolbar, SWT.NONE);
        recentLabel.setText("Recent Files:");
        recentLabel.setFont(boldFont);

        // Recent files combo
        Combo recentCombo = new Combo(toolbar, SWT.DROP_DOWN | SWT.READ_ONLY);
        recentCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        updateRecentFilesCombo(recentCombo);
        recentCombo.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                int index = recentCombo.getSelectionIndex();
                if (index >= 0 && index < recentFiles.size()) {
                    loadDiagramFromPath(recentFiles.get(index));
                }
            }
        });

        // Store combo reference for updates
        this.recentFilesCombo = recentCombo;

        // Separator before Restart
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Run button
        Button runBtn = new Button(toolbar, SWT.PUSH);
        runBtn.setText("Run Pipeline");
        runBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        runBtn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                executePipeline();
            }
        });

        // Restart button
        Button restartBtn = new Button(toolbar, SWT.PUSH);
        restartBtn.setText("Restart");
        restartBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        restartBtn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                restartApplication();
            }
        });
    }

    private void saveDiagramAs() {
        FileDialog dialog = new FileDialog(shell, SWT.SAVE);
        dialog.setText("Save Diagram");
        dialog.setFilterExtensions(new String[]{"*.json"});
        dialog.setFilterNames(new String[]{"JSON Files"});
        dialog.setOverwrite(true);

        String path = dialog.open();
        if (path != null) {
            if (!path.endsWith(".json")) {
                path += ".json";
            }
            saveDiagramToPath(path);
        }
    }

    private void saveDiagramToPath(String path) {
        try {
            JsonObject root = new JsonObject();

            // Save nodes
            JsonArray nodesArray = new JsonArray();
            for (int i = 0; i < nodes.size(); i++) {
                PipelineNode node = nodes.get(i);
                JsonObject nodeObj = new JsonObject();
                nodeObj.addProperty("id", i);
                nodeObj.addProperty("x", node.x);
                nodeObj.addProperty("y", node.y);

                if (node instanceof ImageSourceNode) {
                    nodeObj.addProperty("type", "ImageSource");
                    ImageSourceNode isn = (ImageSourceNode) node;
                    if (isn.imagePath != null) {
                        nodeObj.addProperty("imagePath", isn.imagePath);
                    }
                } else if (node instanceof ProcessingNode) {
                    nodeObj.addProperty("type", "Processing");
                    nodeObj.addProperty("name", ((ProcessingNode) node).getName());
                }

                nodesArray.add(nodeObj);
            }
            root.add("nodes", nodesArray);

            // Save connections
            JsonArray connsArray = new JsonArray();
            for (Connection conn : connections) {
                JsonObject connObj = new JsonObject();
                connObj.addProperty("sourceId", nodes.indexOf(conn.source));
                connObj.addProperty("targetId", nodes.indexOf(conn.target));
                connsArray.add(connObj);
            }
            root.add("connections", connsArray);

            // Write to file
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter writer = new FileWriter(path)) {
                gson.toJson(root, writer);
            }

            currentFilePath = path;
            addToRecentFiles(path);
            shell.setText("OpenCV Pipeline Editor - " + new File(path).getName());

        } catch (Exception e) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_ERROR | SWT.OK);
            mb.setText("Save Error");
            mb.setMessage("Failed to save: " + e.getMessage());
            mb.open();
        }
    }

    private void loadDiagram() {
        FileDialog dialog = new FileDialog(shell, SWT.OPEN);
        dialog.setText("Load Diagram");
        dialog.setFilterExtensions(new String[]{"*.json"});
        dialog.setFilterNames(new String[]{"JSON Files"});

        String path = dialog.open();
        if (path != null) {
            try {
                // Clear existing
                for (PipelineNode node : nodes) {
                    if (node instanceof ImageSourceNode) {
                        ((ImageSourceNode) node).overlayComposite.dispose();
                    }
                }
                nodes.clear();
                connections.clear();

                // Load JSON
                Gson gson = new Gson();
                JsonObject root;
                try (FileReader reader = new FileReader(path)) {
                    root = gson.fromJson(reader, JsonObject.class);
                }

                // Load nodes
                JsonArray nodesArray = root.getAsJsonArray("nodes");
                for (JsonElement elem : nodesArray) {
                    JsonObject nodeObj = elem.getAsJsonObject();
                    int x = nodeObj.get("x").getAsInt();
                    int y = nodeObj.get("y").getAsInt();
                    String type = nodeObj.get("type").getAsString();

                    if ("ImageSource".equals(type)) {
                        ImageSourceNode node = new ImageSourceNode(shell, display, canvas, x, y);
                        if (nodeObj.has("imagePath")) {
                            String imgPath = nodeObj.get("imagePath").getAsString();
                            node.imagePath = imgPath;
                            node.loadImage(imgPath);
                        }
                        nodes.add(node);
                    } else if ("Processing".equals(type)) {
                        String name = nodeObj.get("name").getAsString();
                        ProcessingNode node = new ProcessingNode(display, name, x, y);
                        nodes.add(node);
                    }
                }

                // Load connections
                JsonArray connsArray = root.getAsJsonArray("connections");
                for (JsonElement elem : connsArray) {
                    JsonObject connObj = elem.getAsJsonObject();
                    int sourceId = connObj.get("sourceId").getAsInt();
                    int targetId = connObj.get("targetId").getAsInt();
                    if (sourceId >= 0 && sourceId < nodes.size() &&
                        targetId >= 0 && targetId < nodes.size()) {
                        connections.add(new Connection(nodes.get(sourceId), nodes.get(targetId)));
                    }
                }

                currentFilePath = path;
                addToRecentFiles(path);
                canvas.redraw();
                shell.setText("OpenCV Pipeline Editor - " + new File(path).getName());

            } catch (Exception e) {
                MessageBox mb = new MessageBox(shell, SWT.ICON_ERROR | SWT.OK);
                mb.setText("Load Error");
                mb.setMessage("Failed to load: " + e.getMessage());
                mb.open();
            }
        }
    }

    private void createNodeButton(Composite parent, String text, Runnable action) {
        Button btn = new Button(parent, SWT.PUSH);
        btn.setText(text);
        btn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        btn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                action.run();
            }
        });
    }

    private void createCanvas() {
        canvas = new Canvas(shell, SWT.BORDER | SWT.DOUBLE_BUFFERED);
        canvas.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        canvas.setBackground(display.getSystemColor(SWT.COLOR_WHITE));

        // Paint handler
        canvas.addPaintListener(e -> paintCanvas(e.gc));

        // Mouse handling
        canvas.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseDown(MouseEvent e) {
                handleMouseDown(e);
            }

            @Override
            public void mouseUp(MouseEvent e) {
                handleMouseUp(e);
            }
        });

        canvas.addMouseMoveListener(e -> handleMouseMove(e));

        // Right-click for connections
        canvas.addMenuDetectListener(e -> handleRightClick(e));
    }

    private void createPreviewPanel() {
        Composite previewPanel = new Composite(shell, SWT.BORDER);
        GridData gd = new GridData(SWT.FILL, SWT.FILL, false, true);
        gd.widthHint = 300;
        previewPanel.setLayoutData(gd);
        previewPanel.setLayout(new GridLayout(1, false));

        Label titleLabel = new Label(previewPanel, SWT.NONE);
        titleLabel.setText("Output Preview");
        Font boldFont = new Font(display, "Arial", 11, SWT.BOLD);
        titleLabel.setFont(boldFont);

        previewCanvas = new Canvas(previewPanel, SWT.BORDER | SWT.DOUBLE_BUFFERED);
        previewCanvas.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        previewCanvas.setBackground(display.getSystemColor(SWT.COLOR_GRAY));

        previewCanvas.addPaintListener(e -> {
            if (previewImage != null && !previewImage.isDisposed()) {
                Rectangle bounds = previewImage.getBounds();
                Rectangle canvasBounds = previewCanvas.getClientArea();

                // Scale to fit while maintaining aspect ratio
                double scale = Math.min(
                    (double) canvasBounds.width / bounds.width,
                    (double) canvasBounds.height / bounds.height
                );
                int scaledWidth = (int) (bounds.width * scale);
                int scaledHeight = (int) (bounds.height * scale);
                int x = (canvasBounds.width - scaledWidth) / 2;
                int y = (canvasBounds.height - scaledHeight) / 2;

                e.gc.drawImage(previewImage, 0, 0, bounds.width, bounds.height,
                    x, y, scaledWidth, scaledHeight);
            } else {
                e.gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
                e.gc.drawString("No output yet", 10, 10, true);
                e.gc.drawString("Click 'Run Pipeline'", 10, 30, true);
            }
        });
    }

    private void executePipeline() {
        // Find all nodes and build execution order
        // For now, simple linear execution following connections

        // Find source node (ImageSourceNode with no incoming connections)
        PipelineNode sourceNode = null;
        for (PipelineNode node : nodes) {
            if (node instanceof ImageSourceNode) {
                boolean hasIncoming = false;
                for (Connection conn : connections) {
                    if (conn.target == node) {
                        hasIncoming = true;
                        break;
                    }
                }
                if (!hasIncoming) {
                    sourceNode = node;
                    break;
                }
            }
        }

        if (sourceNode == null) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
            mb.setText("No Source");
            mb.setMessage("No image source node found in the pipeline.");
            mb.open();
            return;
        }

        ImageSourceNode imgSource = (ImageSourceNode) sourceNode;
        Mat currentMat = imgSource.getLoadedImage();

        if (currentMat == null || currentMat.empty()) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
            mb.setText("No Image");
            mb.setMessage("Please load an image in the source node first.");
            mb.open();
            return;
        }

        // Clone the mat so we don't modify the original
        currentMat = currentMat.clone();

        // Follow connections and execute each node
        PipelineNode currentNode = sourceNode;
        while (currentNode != null) {
            // Find next node
            PipelineNode nextNode = null;
            for (Connection conn : connections) {
                if (conn.source == currentNode) {
                    nextNode = conn.target;
                    break;
                }
            }

            if (nextNode == null) break;

            // Execute the processing
            if (nextNode instanceof ProcessingNode) {
                ProcessingNode procNode = (ProcessingNode) nextNode;
                currentMat = applyProcessing(currentMat, procNode.getName());
            }

            currentNode = nextNode;
        }

        // Update preview
        updatePreview(currentMat);
    }

    private Mat applyProcessing(Mat input, String operation) {
        Mat output = new Mat();

        switch (operation) {
            case "Grayscale":
                if (input.channels() == 3) {
                    Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
                } else {
                    output = input.clone();
                }
                break;

            case "Gaussian Blur":
                Imgproc.GaussianBlur(input, output, new Size(15, 15), 0);
                break;

            case "Threshold":
                Mat gray = input;
                if (input.channels() == 3) {
                    gray = new Mat();
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                }
                Imgproc.threshold(gray, output, 127, 255, Imgproc.THRESH_BINARY);
                break;

            case "Canny Edge":
                Mat grayForCanny = input;
                if (input.channels() == 3) {
                    grayForCanny = new Mat();
                    Imgproc.cvtColor(input, grayForCanny, Imgproc.COLOR_BGR2GRAY);
                }
                Imgproc.Canny(grayForCanny, output, 100, 200);
                break;

            case "Output":
                // Output node just passes through
                output = input.clone();
                break;

            default:
                output = input.clone();
                break;
        }

        return output;
    }

    private void updatePreview(Mat mat) {
        // Dispose old preview image
        if (previewImage != null && !previewImage.isDisposed()) {
            previewImage.dispose();
        }

        // Convert Mat to SWT Image
        Mat rgb = new Mat();
        if (mat.channels() == 3) {
            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB);
        } else if (mat.channels() == 1) {
            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_GRAY2RGB);
        } else {
            rgb = mat;
        }

        int width = rgb.width();
        int height = rgb.height();
        byte[] data = new byte[width * height * 3];
        rgb.get(0, 0, data);

        ImageData imageData = new ImageData(width, height, 24,
            new PaletteData(0xFF0000, 0x00FF00, 0x0000FF));
        imageData.data = data;

        previewImage = new Image(display, imageData);
        previewCanvas.redraw();
    }

    private void paintCanvas(GC gc) {
        gc.setAntialias(SWT.ON);

        // Draw connections
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        gc.setLineWidth(2);
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint();
            Point end = conn.target.getInputPoint();
            gc.drawLine(start.x, start.y, end.x, end.y);

            // Draw arrow
            drawArrow(gc, start, end);
        }

        // Draw connection being made
        if (connectionSource != null && connectionEndPoint != null) {
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLUE));
            Point start = connectionSource.getOutputPoint();
            gc.drawLine(start.x, start.y, connectionEndPoint.x, connectionEndPoint.y);
        }

        // Draw nodes
        for (PipelineNode node : nodes) {
            node.paint(gc);
        }
    }

    private void drawArrow(GC gc, Point start, Point end) {
        double angle = Math.atan2(end.y - start.y, end.x - start.x);
        int arrowSize = 10;

        int x1 = (int) (end.x - arrowSize * Math.cos(angle - Math.PI / 6));
        int y1 = (int) (end.y - arrowSize * Math.sin(angle - Math.PI / 6));
        int x2 = (int) (end.x - arrowSize * Math.cos(angle + Math.PI / 6));
        int y2 = (int) (end.y - arrowSize * Math.sin(angle + Math.PI / 6));

        gc.drawLine(end.x, end.y, x1, y1);
        gc.drawLine(end.x, end.y, x2, y2);
    }

    private void handleMouseDown(MouseEvent e) {
        if (e.button == 1) {
            Point clickPoint = new Point(e.x, e.y);

            for (PipelineNode node : nodes) {
                if (node.containsPoint(clickPoint)) {
                    selectedNode = node;
                    dragOffset = new Point(e.x - node.x, e.y - node.y);
                    isDragging = true;
                    return;
                }
            }
        }
    }

    private void handleMouseUp(MouseEvent e) {
        if (connectionSource != null) {
            Point clickPoint = new Point(e.x, e.y);
            for (PipelineNode node : nodes) {
                if (node != connectionSource && node.containsPoint(clickPoint)) {
                    connections.add(new Connection(connectionSource, node));
                    break;
                }
            }
            connectionSource = null;
            connectionEndPoint = null;
            canvas.redraw();
        }

        isDragging = false;
        selectedNode = null;
    }

    private void handleMouseMove(MouseEvent e) {
        if (isDragging && selectedNode != null) {
            selectedNode.x = e.x - dragOffset.x;
            selectedNode.y = e.y - dragOffset.y;
            canvas.redraw();
        } else if (connectionSource != null) {
            connectionEndPoint = new Point(e.x, e.y);
            canvas.redraw();
        }
    }

    private void handleRightClick(MenuDetectEvent e) {
        Point clickPoint = display.map(null, canvas, new Point(e.x, e.y));

        for (PipelineNode node : nodes) {
            if (node.containsPoint(clickPoint)) {
                connectionSource = node;
                connectionEndPoint = clickPoint;
                return;
            }
        }
    }

    private void createSamplePipeline() {
        addImageSourceNodeAt(50, 100);
        addProcessingNodeAt("Grayscale", 300, 100);
        addProcessingNodeAt("Gaussian Blur", 500, 100);
        addProcessingNodeAt("Output", 700, 100);

        if (nodes.size() >= 4) {
            connections.add(new Connection(nodes.get(0), nodes.get(1)));
            connections.add(new Connection(nodes.get(1), nodes.get(2)));
            connections.add(new Connection(nodes.get(2), nodes.get(3)));
        }
    }

    private void addImageSourceNode() {
        addImageSourceNodeAt(50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addImageSourceNodeAt(int x, int y) {
        ImageSourceNode node = new ImageSourceNode(shell, display, canvas, x, y);
        nodes.add(node);
        canvas.redraw();
    }

    private void addProcessingNode(String name) {
        addProcessingNodeAt(name, 50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addProcessingNodeAt(String name, int x, int y) {
        ProcessingNode node = new ProcessingNode(display, name, x, y);
        nodes.add(node);
        canvas.redraw();
    }

    // Base class for pipeline nodes
    abstract static class PipelineNode {
        protected Display display;
        protected int x, y;
        protected int width = 180;
        protected int height = 60;

        public abstract void paint(GC gc);

        public boolean containsPoint(Point p) {
            return p.x >= x && p.x <= x + width && p.y >= y && p.y <= y + height;
        }

        public Point getOutputPoint() {
            return new Point(x + width, y + height / 2);
        }

        public Point getInputPoint() {
            return new Point(x, y + height / 2);
        }
    }

    // Image source node with file chooser and thumbnail
    static class ImageSourceNode extends PipelineNode {
        private Shell shell;
        private Canvas parentCanvas;
        private String imagePath = null;
        private Image thumbnail = null;
        private Mat loadedImage = null;
        private Composite overlayComposite;

        public ImageSourceNode(Shell shell, Display display, Canvas canvas, int x, int y) {
            this.shell = shell;
            this.display = display;
            this.parentCanvas = canvas;
            this.x = x;
            this.y = y;
            this.height = 120;

            createOverlay();
        }

        private void createOverlay() {
            overlayComposite = new Composite(parentCanvas, SWT.NONE);
            overlayComposite.setLayout(new GridLayout(1, false));
            overlayComposite.setBounds(x + 5, y + 25, width - 10, height - 30);

            Button chooseBtn = new Button(overlayComposite, SWT.PUSH);
            chooseBtn.setText("Choose...");
            chooseBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
            chooseBtn.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    chooseImage();
                }
            });

            Label thumbnailLabel = new Label(overlayComposite, SWT.BORDER | SWT.CENTER);
            GridData gd = new GridData(SWT.FILL, SWT.FILL, true, true);
            gd.heightHint = 50;
            thumbnailLabel.setLayoutData(gd);
            thumbnailLabel.setText("No image");

            // Ensure the overlay is visible
            overlayComposite.moveAbove(null);
            overlayComposite.layout();
        }

        private void chooseImage() {
            FileDialog dialog = new FileDialog(shell, SWT.OPEN);
            dialog.setText("Select Image");
            dialog.setFilterExtensions(new String[]{"*.png;*.jpg;*.jpeg;*.bmp;*.tiff", "*.*"});
            dialog.setFilterNames(new String[]{"Image Files", "All Files"});

            String path = dialog.open();
            if (path != null) {
                imagePath = path;
                loadImage(path);
            }
        }

        private void loadImage(String path) {
            loadedImage = Imgcodecs.imread(path);

            if (loadedImage.empty()) {
                return;
            }

            // Create thumbnail
            Mat resized = new Mat();
            double scale = Math.min(140.0 / loadedImage.width(), 45.0 / loadedImage.height());
            Imgproc.resize(loadedImage, resized,
                new Size(loadedImage.width() * scale, loadedImage.height() * scale));

            if (thumbnail != null) {
                thumbnail.dispose();
            }
            thumbnail = matToSwtImage(resized);

            // Update the label
            Control[] children = overlayComposite.getChildren();
            if (children.length > 1 && children[1] instanceof Label) {
                Label label = (Label) children[1];
                label.setText("");
                label.setImage(thumbnail);
            }
        }

        private Image matToSwtImage(Mat mat) {
            Mat rgb = new Mat();
            if (mat.channels() == 3) {
                Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB);
            } else if (mat.channels() == 1) {
                Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_GRAY2RGB);
            } else {
                rgb = mat;
            }

            int width = rgb.width();
            int height = rgb.height();
            byte[] data = new byte[width * height * 3];
            rgb.get(0, 0, data);

            ImageData imageData = new ImageData(width, height, 24,
                new PaletteData(0xFF0000, 0x00FF00, 0x0000FF));
            imageData.data = data;

            return new Image(display, imageData);
        }

        @Override
        public void paint(GC gc) {
            // Update overlay position
            overlayComposite.setBounds(x + 5, y + 25, width - 10, height - 30);

            // Draw node background
            gc.setBackground(new Color(230, 240, 255));
            gc.fillRoundRectangle(x, y, width, height, 10, 10);

            // Draw border
            gc.setForeground(new Color(0, 0, 139));
            gc.setLineWidth(2);
            gc.drawRoundRectangle(x, y, width, height, 10, 10);

            // Draw title
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
            Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
            gc.setFont(boldFont);
            gc.drawString("Image Source", x + 10, y + 5, true);
            boldFont.dispose();
        }

        public Mat getLoadedImage() {
            return loadedImage;
        }
    }

    // Processing node
    static class ProcessingNode extends PipelineNode {
        private String name;

        public ProcessingNode(Display display, String name, int x, int y) {
            this.display = display;
            this.name = name;
            this.x = x;
            this.y = y;
        }

        @Override
        public void paint(GC gc) {
            // Draw node background
            gc.setBackground(new Color(230, 255, 230));
            gc.fillRoundRectangle(x, y, width, height, 10, 10);

            // Draw border
            gc.setForeground(new Color(0, 100, 0));
            gc.setLineWidth(2);
            gc.drawRoundRectangle(x, y, width, height, 10, 10);

            // Draw title
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
            Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
            gc.setFont(boldFont);
            gc.drawString(name, x + 10, y + 10, true);
            boldFont.dispose();

            // Draw parameters placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(parameters)", x + 10, y + 30, true);
        }

        public String getName() {
            return name;
        }
    }

    // Connection between nodes
    static class Connection {
        PipelineNode source;
        PipelineNode target;

        public Connection(PipelineNode source, PipelineNode target) {
            this.source = source;
            this.target = target;
        }
    }
}
