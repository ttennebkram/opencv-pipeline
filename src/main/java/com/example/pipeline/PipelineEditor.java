package com.example.pipeline;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;
import org.eclipse.swt.custom.SashForm;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.prefs.Preferences;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

public class PipelineEditor {

    private Shell shell;
    private Display display;
    private Canvas canvas;
    private Canvas previewCanvas;
    private SashForm sashForm;
    private Image previewImage;

    private List<PipelineNode> nodes = new ArrayList<>();
    private List<Connection> connections = new ArrayList<>();

    private PipelineNode selectedNode = null;
    private Point dragOffset = null;
    private boolean isDragging = false;

    // Connection drawing state
    private PipelineNode connectionSource = null;
    private Point connectionEndPoint = null;

    // Dangling connections (one end attached, one end free)
    private java.util.List<DanglingConnection> danglingConnections = new java.util.ArrayList<>();
    private java.util.List<ReverseDanglingConnection> reverseDanglingConnections = new java.util.ArrayList<>();
    private java.util.List<FreeConnection> freeConnections = new java.util.ArrayList<>();
    private PipelineNode connectionTarget = null; // For reverse dragging (from target end)

    // Free connection dragging state
    private Point freeConnectionFixedEnd = null; // The end that stays fixed while dragging the other
    private boolean draggingFreeConnectionSource = false; // true if dragging source end, false if dragging target end

    // Selection state
    private Set<PipelineNode> selectedNodes = new HashSet<>();
    private Set<Connection> selectedConnections = new HashSet<>();
    private Point selectionBoxStart = null;
    private Point selectionBoxEnd = null;
    private boolean isSelectionBoxDragging = false;

    // Recent files
    private static final int MAX_RECENT_FILES = 10;
    private static final String RECENT_FILES_KEY = "recentFiles";
    private static final String LAST_FILE_KEY = "lastFile";

    // Thumbnail sizes
    private static final int PROCESSING_NODE_THUMB_WIDTH = 120;
    private static final int PROCESSING_NODE_THUMB_HEIGHT = 80;
    private static final int SOURCE_NODE_THUMB_WIDTH = 280;
    private static final int SOURCE_NODE_THUMB_HEIGHT = 90;

    // Node container sizes (based on thumbnail + padding for title/borders)
    private static final int NODE_WIDTH = PROCESSING_NODE_THUMB_WIDTH + 20;  // thumbnail + 10px padding each side
    private static final int NODE_HEIGHT = PROCESSING_NODE_THUMB_HEIGHT + 40; // thumbnail + 25px title + 15px bottom
    private static final int SOURCE_NODE_HEIGHT = SOURCE_NODE_THUMB_HEIGHT + 32; // thumbnail + 22px title + 10px bottom (tighter)
    private Preferences prefs;
    private List<String> recentFiles = new ArrayList<>();
    private Combo recentFilesCombo;
    private Menu openRecentMenu;

    // Threading state
    private AtomicBoolean pipelineRunning = new AtomicBoolean(false);
    private List<Thread> nodeThreads = new ArrayList<>();
    private List<BlockingQueue<Mat>> queues = new ArrayList<>();
    private Button startStopBtn;

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
        shell.setLayout(new GridLayout(2, false));

        // Initialize preferences and load recent files
        prefs = Preferences.userNodeForPackage(PipelineEditor.class);
        loadRecentFiles();

        // Create menu bar
        createMenuBar();

        // Setup system menu (macOS application menu)
        setupSystemMenu();

        // Left side - toolbar/palette
        createToolbar();

        // Right side - SashForm containing canvas and preview panel
        sashForm = new SashForm(shell, SWT.HORIZONTAL);
        sashForm.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));

        // Center - canvas (inside sashForm)
        createCanvas();

        // Right side - preview panel (inside sashForm)
        createPreviewPanel();

        // Set initial weights (75% canvas, 25% preview)
        sashForm.setWeights(new int[] {75, 25});

        // Load last file or create sample pipeline
        String lastFile = prefs.get(LAST_FILE_KEY, "");
        if (!lastFile.isEmpty() && new File(lastFile).exists()) {
            loadDiagramFromPath(lastFile);
        } else {
            createSamplePipeline();
        }

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
            // Add Restart item at the end (no separator - it's a logical group with Quit)
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
        // Save as last file for next startup
        prefs.put(LAST_FILE_KEY, path);
        updateRecentFilesCombo(recentFilesCombo);
        updateOpenRecentMenu();
    }

    private String getCacheDir(String pipelinePath) {
        // Create cache directory next to the pipeline file
        File pipelineFile = new File(pipelinePath);
        String parentDir = pipelineFile.getParent();
        String baseName = pipelineFile.getName();
        if (baseName.endsWith(".json")) {
            baseName = baseName.substring(0, baseName.length() - 5);
        }
        return parentDir + File.separator + "." + baseName + "_cache";
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
            danglingConnections.clear();
            reverseDanglingConnections.clear();
            freeConnections.clear();

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
                        node.loadMedia(imgPath);
                    }
                    if (nodeObj.has("fpsMode")) {
                        node.setFpsMode(nodeObj.get("fpsMode").getAsInt());
                    }
                    nodes.add(node);
                } else if ("Processing".equals(type)) {
                    String name = nodeObj.get("name").getAsString();
                    ProcessingNode node = createEffectNode(name, x, y);
                    if (node != null) {
                        // Load node-specific properties
                        if (node instanceof GaussianBlurNode) {
                            GaussianBlurNode gbn = (GaussianBlurNode) node;
                            if (nodeObj.has("kernelSizeX")) gbn.kernelSizeX = nodeObj.get("kernelSizeX").getAsInt();
                            if (nodeObj.has("kernelSizeY")) gbn.kernelSizeY = nodeObj.get("kernelSizeY").getAsInt();
                            if (nodeObj.has("sigmaX")) gbn.sigmaX = nodeObj.get("sigmaX").getAsDouble();
                        } else if (node instanceof GrayscaleNode) {
                            GrayscaleNode gn = (GrayscaleNode) node;
                            if (nodeObj.has("conversionIndex")) {
                                gn.conversionIndex = nodeObj.get("conversionIndex").getAsInt();
                            }
                        } else if (node instanceof ThresholdNode) {
                            ThresholdNode tn = (ThresholdNode) node;
                            if (nodeObj.has("threshValue")) tn.threshValue = nodeObj.get("threshValue").getAsInt();
                            if (nodeObj.has("maxValue")) tn.maxValue = nodeObj.get("maxValue").getAsInt();
                            if (nodeObj.has("typeIndex")) tn.typeIndex = nodeObj.get("typeIndex").getAsInt();
                            if (nodeObj.has("modifierIndex")) tn.modifierIndex = nodeObj.get("modifierIndex").getAsInt();
                        } else if (node instanceof GainNode) {
                            GainNode gn = (GainNode) node;
                            if (nodeObj.has("gain")) gn.gain = nodeObj.get("gain").getAsDouble();
                        }
                        // InvertNode has no properties to load
                        node.setOnChanged(() -> executePipeline());
                        nodes.add(node);
                    }
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

            // Load dangling connections
            if (root.has("danglingConnections")) {
                JsonArray danglingArray = root.getAsJsonArray("danglingConnections");
                for (JsonElement elem : danglingArray) {
                    JsonObject dangObj = elem.getAsJsonObject();
                    int sourceId = dangObj.get("sourceId").getAsInt();
                    int freeEndX = dangObj.get("freeEndX").getAsInt();
                    int freeEndY = dangObj.get("freeEndY").getAsInt();
                    if (sourceId >= 0 && sourceId < nodes.size()) {
                        DanglingConnection dc = new DanglingConnection(nodes.get(sourceId), new Point(freeEndX, freeEndY));
                        danglingConnections.add(dc);
                    }
                }
            }

            // Load reverse dangling connections
            if (root.has("reverseDanglingConnections")) {
                JsonArray reverseArray = root.getAsJsonArray("reverseDanglingConnections");
                for (JsonElement elem : reverseArray) {
                    JsonObject revObj = elem.getAsJsonObject();
                    int targetId = revObj.get("targetId").getAsInt();
                    int freeEndX = revObj.get("freeEndX").getAsInt();
                    int freeEndY = revObj.get("freeEndY").getAsInt();
                    if (targetId >= 0 && targetId < nodes.size()) {
                        ReverseDanglingConnection rdc = new ReverseDanglingConnection(nodes.get(targetId), new Point(freeEndX, freeEndY));
                        reverseDanglingConnections.add(rdc);
                    }
                }
            }

            // Load free connections
            if (root.has("freeConnections")) {
                JsonArray freeArray = root.getAsJsonArray("freeConnections");
                for (JsonElement elem : freeArray) {
                    JsonObject freeObj = elem.getAsJsonObject();
                    int startEndX = freeObj.get("startEndX").getAsInt();
                    int startEndY = freeObj.get("startEndY").getAsInt();
                    int arrowEndX = freeObj.get("arrowEndX").getAsInt();
                    int arrowEndY = freeObj.get("arrowEndY").getAsInt();
                    FreeConnection fc = new FreeConnection(new Point(startEndX, startEndY), new Point(arrowEndX, arrowEndY));
                    freeConnections.add(fc);
                }
            }

            currentFilePath = path;
            addToRecentFiles(path);

            // Execute pipeline after loading to generate ProcessingNode thumbnails
            executePipeline();

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
        danglingConnections.clear();
        reverseDanglingConnections.clear();
        freeConnections.clear();
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

        createNodeButton(toolbar, "Gaussian Blur", () -> addEffectNode("GaussianBlur"));
        createNodeButton(toolbar, "Threshold", () -> addEffectNode("Threshold"));
        createNodeButton(toolbar, "Color Convert", () -> addEffectNode("Grayscale"));
        createNodeButton(toolbar, "Invert", () -> addEffectNode("Invert"));
        createNodeButton(toolbar, "Gain", () -> addEffectNode("Gain"));
        createNodeButton(toolbar, "Canny Edge", () -> addEffectNode("CannyEdge"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Outputs
        Label outputsLabel = new Label(toolbar, SWT.NONE);
        outputsLabel.setText("Outputs:");
        outputsLabel.setFont(boldFont);

        // createNodeButton(toolbar, "Output", () -> addEffectNode("Output"));

        // Separator
        Label sep = new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL);
        sep.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));


        // Instructions
        Label instructions = new Label(toolbar, SWT.WRAP);
        instructions.setText("Instructions:\n\n" +
            "• Drag nodes to move\n" +
            "• Right-click to connect\n" +
            "• Double-click for properties\n" +
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
        recentLabel.setText("Recent Pipelines:");
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

        // Start/Stop button for continuous threaded execution
        startStopBtn = new Button(toolbar, SWT.PUSH);
        startStopBtn.setText("Start Pipeline");
        startStopBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        startStopBtn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                if (pipelineRunning.get()) {
                    stopPipeline();
                } else {
                    startPipeline();
                }
            }
        });

        // Run once button
        Button runBtn = new Button(toolbar, SWT.PUSH);
        runBtn.setText("Run Once");
        runBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        runBtn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                executePipeline();
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
                    nodeObj.addProperty("fpsMode", isn.getFpsMode());
                } else if (node instanceof ProcessingNode) {
                    nodeObj.addProperty("type", "Processing");
                    nodeObj.addProperty("name", ((ProcessingNode) node).getName());

                    // Save node-specific properties
                    if (node instanceof GaussianBlurNode) {
                        GaussianBlurNode gbn = (GaussianBlurNode) node;
                        nodeObj.addProperty("kernelSizeX", gbn.kernelSizeX);
                        nodeObj.addProperty("kernelSizeY", gbn.kernelSizeY);
                        nodeObj.addProperty("sigmaX", gbn.sigmaX);
                    } else if (node instanceof GrayscaleNode) {
                        GrayscaleNode gn = (GrayscaleNode) node;
                        nodeObj.addProperty("conversionIndex", gn.conversionIndex);
                    } else if (node instanceof ThresholdNode) {
                        ThresholdNode tn = (ThresholdNode) node;
                        nodeObj.addProperty("threshValue", tn.threshValue);
                        nodeObj.addProperty("maxValue", tn.maxValue);
                        nodeObj.addProperty("typeIndex", tn.typeIndex);
                        nodeObj.addProperty("modifierIndex", tn.modifierIndex);
                    } else if (node instanceof GainNode) {
                        GainNode gn = (GainNode) node;
                        nodeObj.addProperty("gain", gn.gain);
                    }
                    // InvertNode has no properties to save
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

            // Save dangling connections (source node attached, free end)
            JsonArray danglingArray = new JsonArray();
            for (DanglingConnection dc : danglingConnections) {
                JsonObject dcObj = new JsonObject();
                dcObj.addProperty("sourceId", nodes.indexOf(dc.source));
                dcObj.addProperty("freeEndX", dc.freeEnd.x);
                dcObj.addProperty("freeEndY", dc.freeEnd.y);
                danglingArray.add(dcObj);
            }
            root.add("danglingConnections", danglingArray);

            // Save reverse dangling connections (target node attached, free end)
            JsonArray reverseDanglingArray = new JsonArray();
            for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
                JsonObject rdcObj = new JsonObject();
                rdcObj.addProperty("targetId", nodes.indexOf(rdc.target));
                rdcObj.addProperty("freeEndX", rdc.freeEnd.x);
                rdcObj.addProperty("freeEndY", rdc.freeEnd.y);
                reverseDanglingArray.add(rdcObj);
            }
            root.add("reverseDanglingConnections", reverseDanglingArray);

            // Save free connections (both ends free)
            JsonArray freeArray = new JsonArray();
            for (FreeConnection fc : freeConnections) {
                JsonObject fcObj = new JsonObject();
                fcObj.addProperty("startEndX", fc.startEnd.x);
                fcObj.addProperty("startEndY", fc.startEnd.y);
                fcObj.addProperty("arrowEndX", fc.arrowEnd.x);
                fcObj.addProperty("arrowEndY", fc.arrowEnd.y);
                freeArray.add(fcObj);
            }
            root.add("freeConnections", freeArray);

            System.out.println("DEBUG save: Saving " + connections.size() + " connections, " +
                danglingConnections.size() + " dangling, " +
                reverseDanglingConnections.size() + " reverse dangling, " +
                freeConnections.size() + " free connections to " + path);

            // Write to file
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter writer = new FileWriter(path)) {
                gson.toJson(root, writer);
            }

            System.out.println("DEBUG save: File written successfully");

            // Save thumbnails to cache directory
            String cacheDir = getCacheDir(path);
            for (PipelineNode node : nodes) {
                if (node instanceof ImageSourceNode) {
                    ((ImageSourceNode) node).saveThumbnailToCache(cacheDir);
                }
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
                danglingConnections.clear();
                reverseDanglingConnections.clear();
                freeConnections.clear();

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
                            System.out.println("DEBUG load: Found imagePath in JSON: " + imgPath);
                            node.imagePath = imgPath;
                            // Load the media (creates thumbnail and loads image for execution)
                            node.loadMedia(imgPath);
                        } else {
                            System.out.println("DEBUG load: No imagePath in JSON for ImageSource node");
                        }
                        nodes.add(node);
                    } else if ("Processing".equals(type)) {
                        String name = nodeObj.get("name").getAsString();
                        ProcessingNode node = createEffectNode(name, x, y);
                        if (node != null) {
                            // Load node-specific properties
                            if (node instanceof GaussianBlurNode) {
                                GaussianBlurNode gbn = (GaussianBlurNode) node;
                                if (nodeObj.has("kernelSizeX")) gbn.kernelSizeX = nodeObj.get("kernelSizeX").getAsInt();
                                if (nodeObj.has("kernelSizeY")) gbn.kernelSizeY = nodeObj.get("kernelSizeY").getAsInt();
                                if (nodeObj.has("sigmaX")) gbn.sigmaX = nodeObj.get("sigmaX").getAsDouble();
                            } else if (node instanceof GrayscaleNode) {
                                GrayscaleNode gn = (GrayscaleNode) node;
                                if (nodeObj.has("conversionIndex")) {
                                    int loadedIndex = nodeObj.get("conversionIndex").getAsInt();
                                    gn.conversionIndex = loadedIndex;
                                    System.out.println("DEBUG load: Set GrayscaleNode conversionIndex to " + loadedIndex + " for node=" + System.identityHashCode(gn));
                                } else {
                                    System.out.println("DEBUG load: No conversionIndex in JSON for GrayscaleNode");
                                }
                            } else if (node instanceof ThresholdNode) {
                                ThresholdNode tn = (ThresholdNode) node;
                                if (nodeObj.has("threshValue")) tn.threshValue = nodeObj.get("threshValue").getAsInt();
                                if (nodeObj.has("maxValue")) tn.maxValue = nodeObj.get("maxValue").getAsInt();
                                if (nodeObj.has("typeIndex")) tn.typeIndex = nodeObj.get("typeIndex").getAsInt();
                                if (nodeObj.has("modifierIndex")) tn.modifierIndex = nodeObj.get("modifierIndex").getAsInt();
                            } else if (node instanceof GainNode) {
                                GainNode gn = (GainNode) node;
                                if (nodeObj.has("gain")) gn.gain = nodeObj.get("gain").getAsDouble();
                            }
                            // InvertNode has no properties to load
                            node.setOnChanged(() -> executePipeline());
                            nodes.add(node);
                        }
                    }
                }

                // Load connections
                if (root.has("connections")) {
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
                }

                // Load dangling connections (source node attached, free end)
                if (root.has("danglingConnections")) {
                    JsonArray danglingArray = root.getAsJsonArray("danglingConnections");
                    for (JsonElement elem : danglingArray) {
                        JsonObject dcObj = elem.getAsJsonObject();
                        int sourceId = dcObj.get("sourceId").getAsInt();
                        int freeEndX = dcObj.get("freeEndX").getAsInt();
                        int freeEndY = dcObj.get("freeEndY").getAsInt();
                        if (sourceId >= 0 && sourceId < nodes.size()) {
                            danglingConnections.add(new DanglingConnection(nodes.get(sourceId), new Point(freeEndX, freeEndY)));
                        }
                    }
                }

                // Load reverse dangling connections (target node attached, free end)
                if (root.has("reverseDanglingConnections")) {
                    JsonArray reverseDanglingArray = root.getAsJsonArray("reverseDanglingConnections");
                    for (JsonElement elem : reverseDanglingArray) {
                        JsonObject rdcObj = elem.getAsJsonObject();
                        int targetId = rdcObj.get("targetId").getAsInt();
                        int freeEndX = rdcObj.get("freeEndX").getAsInt();
                        int freeEndY = rdcObj.get("freeEndY").getAsInt();
                        if (targetId >= 0 && targetId < nodes.size()) {
                            reverseDanglingConnections.add(new ReverseDanglingConnection(nodes.get(targetId), new Point(freeEndX, freeEndY)));
                        }
                    }
                }

                // Load free connections (both ends free)
                if (root.has("freeConnections")) {
                    JsonArray freeArray = root.getAsJsonArray("freeConnections");
                    for (JsonElement elem : freeArray) {
                        JsonObject fcObj = elem.getAsJsonObject();
                        int startEndX = fcObj.get("startEndX").getAsInt();
                        int startEndY = fcObj.get("startEndY").getAsInt();
                        int arrowEndX = fcObj.get("arrowEndX").getAsInt();
                        int arrowEndY = fcObj.get("arrowEndY").getAsInt();
                        freeConnections.add(new FreeConnection(new Point(startEndX, startEndY), new Point(arrowEndX, arrowEndY)));
                    }
                }

                System.out.println("DEBUG load: Loaded " + connections.size() + " connections, " +
                    danglingConnections.size() + " dangling, " +
                    reverseDanglingConnections.size() + " reverse dangling, " +
                    freeConnections.size() + " free connections from " + path);

                currentFilePath = path;
                addToRecentFiles(path);

                // Execute pipeline after loading to generate ProcessingNode thumbnails
                executePipeline();

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
        canvas = new Canvas(sashForm, SWT.BORDER | SWT.DOUBLE_BUFFERED);
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

            @Override
            public void mouseDoubleClick(MouseEvent e) {
                handleDoubleClick(e);
            }
        });

        canvas.addMouseMoveListener(e -> handleMouseMove(e));

        // Keyboard shortcuts
        canvas.addKeyListener(new org.eclipse.swt.events.KeyAdapter() {
            @Override
            public void keyPressed(org.eclipse.swt.events.KeyEvent e) {
                // Cmd-A to select all nodes
                if (e.character == 'a' && (e.stateMask & SWT.MOD1) != 0) {
                    selectedNodes.addAll(nodes);
                    canvas.redraw();
                }
            }
        });

        // Right-click for connections
        canvas.addMenuDetectListener(e -> handleRightClick(e));
    }

    private void createPreviewPanel() {
        Composite previewPanel = new Composite(sashForm, SWT.BORDER);
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
            System.out.println("DEBUG executePipeline: No image loaded in source node, skipping execution");
            return;
        }
        System.out.println("DEBUG executePipeline: Starting execution with image " + currentMat.width() + "x" + currentMat.height());

        // Clone the mat so we don't modify the original
        currentMat = currentMat.clone();

        // Set output on source node
        sourceNode.setOutputMat(currentMat);

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
                System.out.println("DEBUG executePipeline: Processing node " + procNode.getName());
                // Use the node's process method instead of hardcoded switch
                currentMat = procNode.process(currentMat);
                // Set output on this node for thumbnail
                nextNode.setOutputMat(currentMat);
                System.out.println("DEBUG executePipeline: Set output mat for " + procNode.getName() + ", size=" + currentMat.width() + "x" + currentMat.height());
            }

            currentNode = nextNode;
        }

        // Update preview and redraw canvas to show thumbnails
        updatePreview(currentMat);
        canvas.redraw();
    }

    private void startPipeline() {
        if (pipelineRunning.get()) return;

        // Find source node
        ImageSourceNode sourceNode = null;
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
                    sourceNode = (ImageSourceNode) node;
                    break;
                }
            }
        }

        if (sourceNode == null || sourceNode.getLoadedImage() == null) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
            mb.setText("No Source");
            mb.setMessage("Please load an image in the source node first.");
            mb.open();
            return;
        }

        // Build ordered list of nodes following connections
        List<PipelineNode> orderedNodes = new ArrayList<>();
        orderedNodes.add(sourceNode);
        PipelineNode current = sourceNode;
        while (current != null) {
            PipelineNode next = null;
            for (Connection conn : connections) {
                if (conn.source == current) {
                    next = conn.target;
                    orderedNodes.add(next);
                    break;
                }
            }
            current = next;
        }

        if (orderedNodes.size() < 2) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
            mb.setText("No Pipeline");
            mb.setMessage("Connect at least one processing node to the source.");
            mb.open();
            return;
        }

        // Clear old state
        stopPipeline();

        // Create queues between nodes (N-1 queues for N nodes)
        queues.clear();
        for (int i = 0; i < orderedNodes.size() - 1; i++) {
            queues.add(new LinkedBlockingQueue<>(3)); // Capacity 3 for backpressure
        }

        pipelineRunning.set(true);
        final ImageSourceNode finalSource = sourceNode;
        final long frameDelayMs = (long) (1000.0 / finalSource.getFps());

        // Create threads for each node
        for (int i = 0; i < orderedNodes.size(); i++) {
            final int index = i;
            final PipelineNode node = orderedNodes.get(i);
            final BlockingQueue<Mat> inputQueue = (i > 0) ? queues.get(i - 1) : null;
            final BlockingQueue<Mat> outputQueue = (i < queues.size()) ? queues.get(i) : null;

            Thread t = new Thread(() -> {
                int basePriority = Thread.currentThread().getPriority();
                int currentPriority = basePriority;
                try {
                    while (pipelineRunning.get() && !Thread.currentThread().isInterrupted()) {
                        Mat inputMat = null;
                        Mat outputMat = null;

                        if (node instanceof ImageSourceNode) {
                            // Source node: get next frame (video or static image)
                            inputMat = finalSource.getNextFrame();
                            if (inputMat == null) {
                                Thread.sleep(frameDelayMs);
                                continue;
                            }
                            outputMat = inputMat;
                        } else {
                            // Processing node: read from input queue
                            inputMat = inputQueue.take();
                            if (node instanceof ProcessingNode) {
                                ProcessingNode pn = (ProcessingNode) node;
                                outputMat = pn.process(inputMat);
                                inputMat.release(); // Release input after processing
                            } else {
                                outputMat = inputMat;
                            }
                        }

                        // Update thumbnail on UI thread
                        final Mat thumbMat = outputMat.clone();
                        display.asyncExec(() -> {
                            node.setOutputMat(thumbMat);
                            canvas.redraw();

                            // Update preview if this is the last node
                            if (outputQueue == null) {
                                updatePreview(thumbMat);
                            }
                        });

                        // Pass to next node
                        if (outputQueue != null) {
                            // Adaptive priority: adjust by 1 increment at a time
                            int queueSize = outputQueue.size();

                            if (queueSize >= 1) {
                                // Queue backing up - lower priority by 1
                                if (currentPriority > Thread.MIN_PRIORITY) {
                                    currentPriority--;
                                    Thread.currentThread().setPriority(currentPriority);
                                }
                            } else {
                                // Queue empty - raise priority by 1 toward base
                                if (currentPriority < basePriority) {
                                    currentPriority++;
                                    Thread.currentThread().setPriority(currentPriority);
                                }
                            }

                            outputQueue.put(outputMat);
                        } else {
                            // Last node - cleanup
                            outputMat.release();
                        }

                        // Throttle source node based on video FPS
                        if (node instanceof ImageSourceNode) {
                            Thread.sleep(frameDelayMs);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            // Set priority based on node type
            if (node instanceof ImageSourceNode) {
                t.setPriority(Thread.NORM_PRIORITY + 2); // Higher for source
            } else {
                t.setPriority(Thread.NORM_PRIORITY - 1); // Lower for processing
            }
            t.setDaemon(true);
            t.setName("Pipeline-" + node.getClass().getSimpleName() + "-" + i);
            nodeThreads.add(t);
        }

        // Start all threads
        for (Thread t : nodeThreads) {
            t.start();
        }

        // Update button
        startStopBtn.setText("Stop Pipeline");
    }

    private void stopPipeline() {
        pipelineRunning.set(false);

        // Interrupt all threads
        for (Thread t : nodeThreads) {
            t.interrupt();
        }

        // Wait for threads to finish (with timeout)
        for (Thread t : nodeThreads) {
            try {
                t.join(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Clear queues
        for (BlockingQueue<Mat> queue : queues) {
            Mat m;
            while ((m = queue.poll()) != null) {
                m.release();
            }
        }

        nodeThreads.clear();
        queues.clear();

        // Update button
        if (startStopBtn != null && !startStopBtn.isDisposed()) {
            startStopBtn.setText("Start Pipeline");
        }
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

        // Draw nodes first (so connections appear on top)
        for (PipelineNode node : nodes) {
            node.paint(gc);
        }

        // Draw connections on top of nodes
        gc.setLineWidth(2);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint();
            Point end = conn.target.getInputPoint();
            gc.drawLine(start.x, start.y, end.x, end.y);
            drawArrow(gc, start, end);
        }

        // Draw dangling connections with dashed lines
        gc.setLineStyle(SWT.LINE_DASH);
        gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
        for (DanglingConnection dangling : danglingConnections) {
            Point start = dangling.source.getOutputPoint();
            gc.drawLine(start.x, start.y, dangling.freeEnd.x, dangling.freeEnd.y);
            drawArrow(gc, start, dangling.freeEnd);
            // Draw small circles at both ends to make them easier to grab
            gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.fillOval(start.x - 4, start.y - 4, 8, 8);
            gc.fillOval(dangling.freeEnd.x - 4, dangling.freeEnd.y - 4, 8, 8);
        }
        // Draw reverse dangling connections (target fixed, source free)
        for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
            Point end = dangling.target.getInputPoint();
            gc.drawLine(dangling.freeEnd.x, dangling.freeEnd.y, end.x, end.y);
            // Draw small circles at both ends to make them easier to grab
            gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.fillOval(dangling.freeEnd.x - 4, dangling.freeEnd.y - 4, 8, 8);
            gc.fillOval(end.x - 4, end.y - 4, 8, 8);
        }
        // Draw free connections (both ends free)
        for (FreeConnection free : freeConnections) {
            gc.drawLine(free.startEnd.x, free.startEnd.y, free.arrowEnd.x, free.arrowEnd.y);
            drawArrow(gc, free.startEnd, free.arrowEnd);
            // Draw small circles at both ends to make them easier to grab
            gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.fillOval(free.startEnd.x - 4, free.startEnd.y - 4, 8, 8);
            gc.fillOval(free.arrowEnd.x - 4, free.arrowEnd.y - 4, 8, 8);
        }
        gc.setLineStyle(SWT.LINE_SOLID);

        // Draw connection being made (from source)
        if (connectionSource != null && connectionEndPoint != null) {
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLUE));
            Point start = connectionSource.getOutputPoint();
            gc.drawLine(start.x, start.y, connectionEndPoint.x, connectionEndPoint.y);
        }
        // Draw connection being made (from target - reverse direction)
        if (connectionTarget != null && connectionEndPoint != null) {
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLUE));
            Point end = connectionTarget.getInputPoint();
            gc.drawLine(connectionEndPoint.x, connectionEndPoint.y, end.x, end.y);
        }
        // Draw free connection being dragged (both ends unattached)
        if (freeConnectionFixedEnd != null && connectionEndPoint != null) {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.setLineStyle(SWT.LINE_DASH);
            if (draggingFreeConnectionSource) {
                // Dragging source end - connectionEndPoint is source, freeConnectionFixedEnd is target (arrow end)
                gc.drawLine(connectionEndPoint.x, connectionEndPoint.y, freeConnectionFixedEnd.x, freeConnectionFixedEnd.y);
                drawArrow(gc, connectionEndPoint, freeConnectionFixedEnd);
                // Draw circles at both ends
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.fillOval(connectionEndPoint.x - 4, connectionEndPoint.y - 4, 8, 8);
                gc.fillOval(freeConnectionFixedEnd.x - 4, freeConnectionFixedEnd.y - 4, 8, 8);
            } else {
                // Dragging target end - freeConnectionFixedEnd is source, connectionEndPoint is target (arrow end)
                gc.drawLine(freeConnectionFixedEnd.x, freeConnectionFixedEnd.y, connectionEndPoint.x, connectionEndPoint.y);
                drawArrow(gc, freeConnectionFixedEnd, connectionEndPoint);
                // Draw circles at both ends
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.fillOval(freeConnectionFixedEnd.x - 4, freeConnectionFixedEnd.y - 4, 8, 8);
                gc.fillOval(connectionEndPoint.x - 4, connectionEndPoint.y - 4, 8, 8);
            }
            gc.setLineStyle(SWT.LINE_SOLID);
        }

        // Draw selection highlights
        for (PipelineNode node : nodes) {
            node.drawSelectionHighlight(gc, selectedNodes.contains(node));
        }

        // Draw selection box if dragging
        if (isSelectionBoxDragging && selectionBoxStart != null && selectionBoxEnd != null) {
            int boxX = Math.min(selectionBoxStart.x, selectionBoxEnd.x);
            int boxY = Math.min(selectionBoxStart.y, selectionBoxEnd.y);
            int boxWidth = Math.abs(selectionBoxEnd.x - selectionBoxStart.x);
            int boxHeight = Math.abs(selectionBoxEnd.y - selectionBoxStart.y);

            // Draw selection box with semi-transparent fill
            gc.setBackground(new Color(0, 120, 215));
            gc.setAlpha(30);
            gc.fillRectangle(boxX, boxY, boxWidth, boxHeight);
            gc.setAlpha(255);

            // Draw selection box border
            gc.setForeground(new Color(0, 120, 215));
            gc.setLineWidth(1);
            gc.drawRectangle(boxX, boxY, boxWidth, boxHeight);
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
            int radius = 8; // Slightly larger than visual for easier clicking
            boolean cmdHeld = (e.stateMask & SWT.MOD1) != 0;

            // Debug: Log state at start of click handling
            System.out.println("DEBUG mouseDown: click at " + clickPoint.x + "," + clickPoint.y +
                             " danglingConnections=" + danglingConnections.size() +
                             " reverseDanglingConnections=" + reverseDanglingConnections.size() +
                             " freeConnections=" + freeConnections.size());

            // First check if clicking on an input connection point (to yank off existing connection)
            for (PipelineNode node : nodes) {
                Point inputPoint = node.getInputPoint();
                double dist = Math.sqrt(Math.pow(clickPoint.x - inputPoint.x, 2) +
                                       Math.pow(clickPoint.y - inputPoint.y, 2));
                if (dist <= radius) {
                    // Check if there's a connection to this input
                    Connection connToRemove = null;
                    for (Connection conn : connections) {
                        if (conn.target == node) {
                            connToRemove = conn;
                            break;
                        }
                    }
                    if (connToRemove != null) {
                        // Yank off the connection - remove it and start dragging from the source
                        System.out.println("DEBUG: Yanking regular connection from input point");
                        connectionSource = connToRemove.source;
                        connectionEndPoint = clickPoint;
                        connections.remove(connToRemove);
                        canvas.redraw();
                        return;
                    }
                    // Check if there's a reverse dangling connection to this input
                    // If so, we should yank it to create a FreeConnection
                    ReverseDanglingConnection reverseToYank = null;
                    for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
                        if (dangling.target == node) {
                            reverseToYank = dangling;
                            break;
                        }
                    }
                    if (reverseToYank != null) {
                        System.out.println("DEBUG: Yanking reverse dangling connection from input point - creating FreeConnection");
                        reverseDanglingConnections.remove(reverseToYank);
                        freeConnections.add(new FreeConnection(reverseToYank.freeEnd, new Point(inputPoint.x, inputPoint.y)));
                        canvas.redraw();
                        return;
                    }
                    // No existing connection - start a new connection from input point (reverse direction)
                    System.out.println("DEBUG: Starting new connection from input point (reverse direction)");
                    connectionTarget = node;
                    connectionEndPoint = clickPoint;
                    canvas.redraw();
                    return;
                }
            }

            // Check if clicking on a dangling connection's free end
            DanglingConnection danglingToRemove = null;
            for (DanglingConnection dangling : danglingConnections) {
                double dist = Math.sqrt(Math.pow(clickPoint.x - dangling.freeEnd.x, 2) +
                                       Math.pow(clickPoint.y - dangling.freeEnd.y, 2));
                if (dist <= radius) {
                    // Pick up this dangling connection
                    connectionSource = dangling.source;
                    connectionEndPoint = clickPoint;
                    danglingToRemove = dangling;
                    break;
                }
            }
            if (danglingToRemove != null) {
                danglingConnections.remove(danglingToRemove);
                canvas.redraw();
                return;
            }

            // Check if clicking on a dangling connection's source end (output point of its source node)
            DanglingConnection danglingSourceToRemove = null;
            Point danglingSourcePoint = null;
            Point danglingFreeEnd = null;
            for (DanglingConnection dangling : danglingConnections) {
                Point sourcePoint = dangling.source.getOutputPoint();
                double dist = Math.sqrt(Math.pow(clickPoint.x - sourcePoint.x, 2) +
                                       Math.pow(clickPoint.y - sourcePoint.y, 2));
                System.out.println("DEBUG handleMouseDown: Checking dangling source end at " + sourcePoint.x + "," + sourcePoint.y + " vs click " + clickPoint.x + "," + clickPoint.y + " dist=" + dist);
                if (dist <= radius) {
                    // Yank off the source end - this becomes a completely free connector
                    System.out.println("DEBUG handleMouseDown: MATCHED dangling source end, creating FreeConnection");
                    danglingSourceToRemove = dangling;
                    danglingSourcePoint = new Point(sourcePoint.x, sourcePoint.y);
                    danglingFreeEnd = dangling.freeEnd;
                    break;
                }
            }
            if (danglingSourceToRemove != null) {
                danglingConnections.remove(danglingSourceToRemove);
                // Instead of just creating a FreeConnection, set up dragging from the source end
                // The target end stays fixed, we drag the source end
                System.out.println("DEBUG handleMouseDown: Yanking source end of DanglingConnection - start dragging with target at " + danglingFreeEnd.x + "," + danglingFreeEnd.y);
                connectionTarget = null; // No target node
                connectionSource = null; // No source node
                connectionEndPoint = clickPoint; // Current drag position
                // Store the fixed end in a temporary way - we need a new state variable
                // For now, create a ReverseDanglingConnection-like drag where we drag the source
                // Actually, let's just use connectionSource = null and connectionTarget = null
                // and create a FreeConnection on mouse up, but allow dragging
                // We need to track the OTHER end of the free connection
                freeConnectionFixedEnd = danglingFreeEnd; // The arrow end stays put
                draggingFreeConnectionSource = true; // We're dragging the source end
                canvas.redraw();
                return;
            }

            // Check if clicking on a reverse dangling connection's free end
            ReverseDanglingConnection reverseDanglingToRemove = null;
            for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
                double dist = Math.sqrt(Math.pow(clickPoint.x - dangling.freeEnd.x, 2) +
                                       Math.pow(clickPoint.y - dangling.freeEnd.y, 2));
                if (dist <= radius) {
                    // Pick up this reverse dangling connection
                    connectionTarget = dangling.target;
                    connectionEndPoint = clickPoint;
                    reverseDanglingToRemove = dangling;
                    break;
                }
            }
            if (reverseDanglingToRemove != null) {
                reverseDanglingConnections.remove(reverseDanglingToRemove);
                canvas.redraw();
                return;
            }

            // Check if clicking on a reverse dangling connection's target end (input point of its target node)
            ReverseDanglingConnection reverseTargetToRemove = null;
            Point reverseTargetPoint = null;
            Point reverseFreeEnd = null;
            for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
                Point targetPoint = dangling.target.getInputPoint();
                double dist = Math.sqrt(Math.pow(clickPoint.x - targetPoint.x, 2) +
                                       Math.pow(clickPoint.y - targetPoint.y, 2));
                if (dist <= radius) {
                    // Yank off the target end - this becomes a completely free connector
                    reverseTargetToRemove = dangling;
                    reverseTargetPoint = new Point(targetPoint.x, targetPoint.y);
                    reverseFreeEnd = dangling.freeEnd;
                    break;
                }
            }
            if (reverseTargetToRemove != null) {
                reverseDanglingConnections.remove(reverseTargetToRemove);
                // Create a free connection with both ends unattached
                freeConnections.add(new FreeConnection(reverseFreeEnd, reverseTargetPoint));
                canvas.redraw();
                return;
            }

            // Check if clicking on a FreeConnection's start end (circle at the beginning)
            FreeConnection freeStartToRemove = null;
            Point freeStartOtherEnd = null;
            for (FreeConnection free : freeConnections) {
                double dist = Math.sqrt(Math.pow(clickPoint.x - free.startEnd.x, 2) +
                                       Math.pow(clickPoint.y - free.startEnd.y, 2));
                if (dist <= radius) {
                    System.out.println("DEBUG handleMouseDown: Picking up FreeConnection start end");
                    freeStartToRemove = free;
                    freeStartOtherEnd = free.arrowEnd;
                    break;
                }
            }
            if (freeStartToRemove != null) {
                freeConnections.remove(freeStartToRemove);
                // Set up dragging from the start end
                freeConnectionFixedEnd = freeStartOtherEnd; // The arrow end stays put
                draggingFreeConnectionSource = true; // We're dragging the start end
                connectionEndPoint = clickPoint;
                canvas.redraw();
                return;
            }

            // Check if clicking on a FreeConnection's arrow end (arrow at the end)
            FreeConnection freeArrowToRemove = null;
            Point freeArrowOtherEnd = null;
            for (FreeConnection free : freeConnections) {
                double dist = Math.sqrt(Math.pow(clickPoint.x - free.arrowEnd.x, 2) +
                                       Math.pow(clickPoint.y - free.arrowEnd.y, 2));
                if (dist <= radius) {
                    System.out.println("DEBUG handleMouseDown: Picking up FreeConnection arrow end");
                    freeArrowToRemove = free;
                    freeArrowOtherEnd = free.startEnd;
                    break;
                }
            }
            if (freeArrowToRemove != null) {
                freeConnections.remove(freeArrowToRemove);
                // Set up dragging from the arrow end
                freeConnectionFixedEnd = freeArrowOtherEnd; // The start end stays put
                draggingFreeConnectionSource = false; // We're dragging the arrow end
                connectionEndPoint = clickPoint;
                canvas.redraw();
                return;
            }

            // Check if clicking on an output connection point (only to yank existing connections)
            for (PipelineNode node : nodes) {
                Point outputPoint = node.getOutputPoint();
                double dist = Math.sqrt(Math.pow(clickPoint.x - outputPoint.x, 2) +
                                       Math.pow(clickPoint.y - outputPoint.y, 2));
                if (dist <= radius) {
                    // Check if there's a connection from this output
                    Connection connToRemove = null;
                    for (Connection conn : connections) {
                        if (conn.source == node) {
                            connToRemove = conn;
                            break;
                        }
                    }
                    if (connToRemove != null) {
                        // Yank off the connection - remove it and start dragging from the target
                        System.out.println("DEBUG: Yanking regular connection from output point");
                        connectionTarget = connToRemove.target;
                        connectionEndPoint = clickPoint;
                        connections.remove(connToRemove);
                        canvas.redraw();
                        return;
                    }
                    // Check if there's a dangling connection from this output
                    // If so, we should yank it to create a FreeConnection
                    DanglingConnection danglingToYank = null;
                    for (DanglingConnection dangling : danglingConnections) {
                        if (dangling.source == node) {
                            danglingToYank = dangling;
                            break;
                        }
                    }
                    if (danglingToYank != null) {
                        System.out.println("DEBUG: Yanking dangling connection from output point - creating FreeConnection");
                        danglingConnections.remove(danglingToYank);
                        freeConnections.add(new FreeConnection(new Point(outputPoint.x, outputPoint.y), danglingToYank.freeEnd));
                        canvas.redraw();
                        return;
                    }
                    // No existing connection - start a new connection from output point
                    connectionSource = node;
                    connectionEndPoint = clickPoint;
                    canvas.redraw();
                    return;
                }
            }

            // Check for node selection and dragging
            for (PipelineNode node : nodes) {
                if (node.containsPoint(clickPoint)) {
                    // Handle selection
                    if (cmdHeld) {
                        // Toggle selection of this node
                        if (selectedNodes.contains(node)) {
                            selectedNodes.remove(node);
                        } else {
                            selectedNodes.add(node);
                        }
                    } else {
                        // If node is not selected, clear selection and select only this node
                        if (!selectedNodes.contains(node)) {
                            selectedNodes.clear();
                            selectedConnections.clear();
                            selectedNodes.add(node);
                        }
                        // If node is already selected, keep current selection (for dragging multiple)
                    }

                    // Start dragging
                    selectedNode = node;
                    dragOffset = new Point(e.x - node.x, e.y - node.y);
                    isDragging = true;
                    canvas.redraw();
                    return;
                }
            }

            // Clicked on empty space - start selection box
            if (!cmdHeld) {
                selectedNodes.clear();
                selectedConnections.clear();
            }
            selectionBoxStart = clickPoint;
            selectionBoxEnd = clickPoint;
            isSelectionBoxDragging = true;
            canvas.redraw();
        }
    }

    private void handleMouseUp(MouseEvent e) {
        if (connectionSource != null) {
            Point clickPoint = new Point(e.x, e.y);
            boolean connected = false;
            PipelineNode targetNode = null;

            // Check if dropped on an input point
            for (PipelineNode node : nodes) {
                if (node != connectionSource) {
                    Point inputPoint = node.getInputPoint();
                    int radius = 8;
                    double dist = Math.sqrt(Math.pow(clickPoint.x - inputPoint.x, 2) +
                                           Math.pow(clickPoint.y - inputPoint.y, 2));
                    if (dist <= radius) {
                        targetNode = node;
                        connected = true;
                        break;
                    }
                }
            }

            // If not on input point, check if on node body as fallback
            if (!connected) {
                for (PipelineNode node : nodes) {
                    if (node != connectionSource && node.containsPoint(clickPoint)) {
                        targetNode = node;
                        connected = true;
                        break;
                    }
                }
            }

            if (connected && targetNode != null) {
                // Create a new connection
                connections.add(new Connection(connectionSource, targetNode));
                System.out.println("DEBUG mouseUp: Created regular connection");
            } else if (connectionEndPoint != null) {
                // Create a dangling connection
                danglingConnections.add(new DanglingConnection(connectionSource, connectionEndPoint));
                System.out.println("DEBUG mouseUp: Created DanglingConnection, source=" + connectionSource + " freeEnd=" + connectionEndPoint.x + "," + connectionEndPoint.y + " total=" + danglingConnections.size());
            } else {
                System.out.println("DEBUG mouseUp: No connectionEndPoint, nothing created");
            }

            connectionSource = null;
            connectionEndPoint = null;
            canvas.redraw();

            if (connected) {
                executePipeline();
            }
        }

        // Handle reverse connection (dragging from target end)
        if (connectionTarget != null) {
            Point clickPoint = new Point(e.x, e.y);
            boolean connected = false;
            PipelineNode sourceNode = null;

            // Check if dropped on an output point
            for (PipelineNode node : nodes) {
                if (node != connectionTarget) {
                    Point outputPoint = node.getOutputPoint();
                    int radius = 8;
                    double dist = Math.sqrt(Math.pow(clickPoint.x - outputPoint.x, 2) +
                                           Math.pow(clickPoint.y - outputPoint.y, 2));
                    if (dist <= radius) {
                        sourceNode = node;
                        connected = true;
                        break;
                    }
                }
            }

            // If not on output point, check if on node body as fallback
            if (!connected) {
                for (PipelineNode node : nodes) {
                    if (node != connectionTarget && node.containsPoint(clickPoint)) {
                        sourceNode = node;
                        connected = true;
                        break;
                    }
                }
            }

            if (connected && sourceNode != null) {
                // Create a new connection
                connections.add(new Connection(sourceNode, connectionTarget));
            } else if (connectionEndPoint != null) {
                // Create a reverse dangling connection
                reverseDanglingConnections.add(new ReverseDanglingConnection(connectionTarget, connectionEndPoint));
            }

            connectionTarget = null;
            connectionEndPoint = null;
            canvas.redraw();

            if (connected) {
                executePipeline();
            }
        }

        // Handle free connection dragging (both ends unattached)
        if (freeConnectionFixedEnd != null) {
            Point clickPoint = new Point(e.x, e.y);
            boolean connected = false;
            int radius = 8;

            if (draggingFreeConnectionSource) {
                // Dragging the source end - check if dropped on output point
                PipelineNode sourceNode = null;
                for (PipelineNode node : nodes) {
                    Point outputPoint = node.getOutputPoint();
                    double dist = Math.sqrt(Math.pow(clickPoint.x - outputPoint.x, 2) +
                                           Math.pow(clickPoint.y - outputPoint.y, 2));
                    if (dist <= radius) {
                        sourceNode = node;
                        connected = true;
                        break;
                    }
                }

                if (connected && sourceNode != null) {
                    // Connected to output point - create DanglingConnection
                    System.out.println("DEBUG mouseUp: Free connection source attached to node, creating DanglingConnection");
                    danglingConnections.add(new DanglingConnection(sourceNode, freeConnectionFixedEnd));
                } else {
                    // Not connected - create FreeConnection at current position
                    System.out.println("DEBUG mouseUp: Free connection source not attached, creating FreeConnection");
                    freeConnections.add(new FreeConnection(connectionEndPoint, freeConnectionFixedEnd));
                }
            } else {
                // Dragging the target end - check if dropped on input point
                PipelineNode targetNode = null;
                for (PipelineNode node : nodes) {
                    Point inputPoint = node.getInputPoint();
                    double dist = Math.sqrt(Math.pow(clickPoint.x - inputPoint.x, 2) +
                                           Math.pow(clickPoint.y - inputPoint.y, 2));
                    if (dist <= radius) {
                        targetNode = node;
                        connected = true;
                        break;
                    }
                }

                if (connected && targetNode != null) {
                    // Connected to input point - create ReverseDanglingConnection
                    System.out.println("DEBUG mouseUp: Free connection target attached to node, creating ReverseDanglingConnection");
                    reverseDanglingConnections.add(new ReverseDanglingConnection(targetNode, freeConnectionFixedEnd));
                } else {
                    // Not connected - create FreeConnection at current position
                    System.out.println("DEBUG mouseUp: Free connection target not attached, creating FreeConnection");
                    freeConnections.add(new FreeConnection(freeConnectionFixedEnd, connectionEndPoint));
                }
            }

            freeConnectionFixedEnd = null;
            draggingFreeConnectionSource = false;
            connectionEndPoint = null;
            canvas.redraw();
        }

        // Handle selection box completion
        if (isSelectionBoxDragging && selectionBoxStart != null && selectionBoxEnd != null) {
            // Calculate box bounds
            int boxX = Math.min(selectionBoxStart.x, selectionBoxEnd.x);
            int boxY = Math.min(selectionBoxStart.y, selectionBoxEnd.y);
            int boxWidth = Math.abs(selectionBoxEnd.x - selectionBoxStart.x);
            int boxHeight = Math.abs(selectionBoxEnd.y - selectionBoxStart.y);

            // Select nodes that are completely surrounded by the box
            for (PipelineNode node : nodes) {
                // Check if node is completely inside selection box
                if (node.x >= boxX && node.y >= boxY &&
                    node.x + node.width <= boxX + boxWidth &&
                    node.y + node.height <= boxY + boxHeight) {
                    selectedNodes.add(node);
                }
            }

            // Select connections that are completely inside selection box
            for (Connection conn : connections) {
                Point start = conn.source.getOutputPoint();
                Point end = conn.target.getInputPoint();
                // Check if both endpoints are inside selection box
                if (start.x >= boxX && start.x <= boxX + boxWidth &&
                    start.y >= boxY && start.y <= boxY + boxHeight &&
                    end.x >= boxX && end.x <= boxX + boxWidth &&
                    end.y >= boxY && end.y <= boxY + boxHeight) {
                    selectedConnections.add(conn);
                }
            }

            // Clear selection box state
            selectionBoxStart = null;
            selectionBoxEnd = null;
            isSelectionBoxDragging = false;
            canvas.redraw();
        }

        isDragging = false;
        selectedNode = null;
    }

    private void handleMouseMove(MouseEvent e) {
        if (isDragging && selectedNode != null) {
            // Calculate the delta movement
            int deltaX = e.x - dragOffset.x - selectedNode.x;
            int deltaY = e.y - dragOffset.y - selectedNode.y;

            // Move all selected nodes by the same delta
            if (selectedNodes.contains(selectedNode) && selectedNodes.size() > 1) {
                for (PipelineNode node : selectedNodes) {
                    node.x += deltaX;
                    node.y += deltaY;
                }
            } else {
                // Single node drag
                selectedNode.x = e.x - dragOffset.x;
                selectedNode.y = e.y - dragOffset.y;
            }
            canvas.redraw();
        } else if (connectionSource != null || connectionTarget != null || freeConnectionFixedEnd != null) {
            connectionEndPoint = new Point(e.x, e.y);
            canvas.redraw();
        } else if (isSelectionBoxDragging) {
            selectionBoxEnd = new Point(e.x, e.y);
            canvas.redraw();
        }
    }

    private void handleDoubleClick(MouseEvent e) {
        Point clickPoint = new Point(e.x, e.y);
        for (PipelineNode node : nodes) {
            if (node.containsPoint(clickPoint)) {
                if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).showPropertiesDialog();
                } else if (node instanceof ImageSourceNode) {
                    ((ImageSourceNode) node).showPropertiesDialog();
                }
                return;
            }
        }
    }

    private void handleRightClick(MenuDetectEvent e) {
        Point clickPoint = display.map(null, canvas, new Point(e.x, e.y));

        for (PipelineNode node : nodes) {
            if (node.containsPoint(clickPoint)) {
                // Show context menu for the node
                Menu contextMenu = new Menu(canvas);

                // Edit Properties option (for ProcessingNode and ImageSourceNode)
                if (node instanceof ProcessingNode) {
                    MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
                    editItem.setText("Edit Properties...");
                    editItem.addListener(SWT.Selection, evt -> {
                        ((ProcessingNode) node).showPropertiesDialog();
                    });

                    new MenuItem(contextMenu, SWT.SEPARATOR);
                } else if (node instanceof ImageSourceNode) {
                    MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
                    editItem.setText("Edit Properties...");
                    editItem.addListener(SWT.Selection, evt -> {
                        ((ImageSourceNode) node).showPropertiesDialog();
                    });

                    new MenuItem(contextMenu, SWT.SEPARATOR);
                }

                // Start Connection option
                MenuItem connectItem = new MenuItem(contextMenu, SWT.PUSH);
                connectItem.setText("Start Connection");
                connectItem.addListener(SWT.Selection, evt -> {
                    connectionSource = node;
                    connectionEndPoint = clickPoint;
                });

                // Delete Node option
                MenuItem deleteItem = new MenuItem(contextMenu, SWT.PUSH);
                deleteItem.setText("Delete Node");
                deleteItem.addListener(SWT.Selection, evt -> {
                    // Remove all connections involving this node
                    connections.removeIf(c -> c.source == node || c.target == node);
                    nodes.remove(node);
                    canvas.redraw();
                    executePipeline();
                });

                contextMenu.setLocation(e.x, e.y);
                contextMenu.setVisible(true);
                return;
            }
        }

        // Check if right-clicked on a connection
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint();
            Point end = conn.target.getInputPoint();
            if (isPointNearLine(clickPoint, start, end, 5)) {
                // Show context menu for the connection
                Menu contextMenu = new Menu(canvas);

                MenuItem deleteItem = new MenuItem(contextMenu, SWT.PUSH);
                deleteItem.setText("Delete Connection");
                deleteItem.addListener(SWT.Selection, evt -> {
                    connections.remove(conn);
                    canvas.redraw();
                    executePipeline();
                });

                contextMenu.setLocation(e.x, e.y);
                contextMenu.setVisible(true);
                return;
            }
        }
    }

    // Check if a point is within a certain distance of a line segment
    private boolean isPointNearLine(Point p, Point lineStart, Point lineEnd, double threshold) {
        double dx = lineEnd.x - lineStart.x;
        double dy = lineEnd.y - lineStart.y;
        double lengthSquared = dx * dx + dy * dy;

        if (lengthSquared == 0) {
            // Line segment is a point
            double dist = Math.sqrt(Math.pow(p.x - lineStart.x, 2) + Math.pow(p.y - lineStart.y, 2));
            return dist <= threshold;
        }

        // Calculate the projection of point p onto the line
        double t = Math.max(0, Math.min(1, ((p.x - lineStart.x) * dx + (p.y - lineStart.y) * dy) / lengthSquared));

        // Find the closest point on the line segment
        double closestX = lineStart.x + t * dx;
        double closestY = lineStart.y + t * dy;

        // Calculate distance from p to the closest point
        double distance = Math.sqrt(Math.pow(p.x - closestX, 2) + Math.pow(p.y - closestY, 2));

        return distance <= threshold;
    }

    private void createSamplePipeline() {
        addImageSourceNodeAt(50, 100);
        addEffectNodeAt("Grayscale", 300, 100);
        addEffectNodeAt("GaussianBlur", 500, 100);
        addEffectNodeAt("Threshold", 700, 100);

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

    private void addEffectNode(String type) {
        addEffectNodeAt(type, 50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addEffectNodeAt(String type, int x, int y) {
        ProcessingNode node = createEffectNode(type, x, y);
        if (node != null) {
            node.setOnChanged(() -> executePipeline());
            nodes.add(node);
            canvas.redraw();
        }
    }

    private ProcessingNode createEffectNode(String type, int x, int y) {
        switch (type) {
            case "GaussianBlur":
            case "Gaussian Blur":  // backward compatibility
            case "Blur":           // backward compatibility
                return new GaussianBlurNode(display, shell, x, y);
            case "Threshold":
            case "Threshold (Simple)":  // backward compatibility
                return new ThresholdNode(display, shell, x, y);
            case "Grayscale":
            case "Color Convert":  // backward compatibility
                return new GrayscaleNode(display, shell, x, y);
            case "Invert":
                return new InvertNode(display, shell, x, y);
            case "Gain":
                return new GainNode(display, shell, x, y);
            case "CannyEdge":
            case "Canny Edge":  // backward compatibility
            case "Edges Canny":  // match Python name
                return new CannyEdgeNode(display, shell, x, y);
            default:
                // For any unknown type, create a default GaussianBlur as placeholder
                System.err.println("Unknown effect type: " + type + ", creating GaussianBlur as placeholder");
                return new GaussianBlurNode(display, shell, x, y);
        }
    }

    // Base class for pipeline nodes
    abstract static class PipelineNode {
        protected Display display;
        protected int x, y;
        protected int width = NODE_WIDTH;
        protected int height = NODE_HEIGHT;
        protected Image thumbnail;
        protected Mat outputMat;

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

        // Draw connection points (circles) on the node - always visible
        protected void drawConnectionPoints(GC gc) {
            int radius = 6;  // Slightly larger for visibility

            // Draw input point on left side (blue tint for input)
            Point input = getInputPoint();
            gc.setBackground(new Color(200, 220, 255));  // Light blue fill
            gc.fillOval(input.x - radius, input.y - radius, radius * 2, radius * 2);
            gc.setForeground(new Color(70, 100, 180));   // Blue border
            gc.setLineWidth(2);
            gc.drawOval(input.x - radius, input.y - radius, radius * 2, radius * 2);

            // Draw output point on right side (orange tint for output)
            Point output = getOutputPoint();
            gc.setBackground(new Color(255, 230, 200)); // Light orange fill
            gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
            gc.setForeground(new Color(200, 120, 50));  // Orange border
            gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
            gc.setLineWidth(1);  // Reset line width
        }

        // Draw selection highlight around node
        protected void drawSelectionHighlight(GC gc, boolean isSelected) {
            if (isSelected) {
                gc.setForeground(new Color(0, 120, 215));  // Blue selection color
                gc.setLineWidth(3);
                gc.drawRoundRectangle(x - 3, y - 3, width + 6, height + 6, 13, 13);
            }
        }

        public void setOutputMat(Mat mat) {
            this.outputMat = mat;
            updateThumbnail();
        }

        protected void updateThumbnail() {
            System.out.println("DEBUG updateThumbnail: Called, outputMat=" + (outputMat != null ? outputMat.width() + "x" + outputMat.height() : "null"));
            if (outputMat == null || outputMat.empty()) {
                System.out.println("DEBUG updateThumbnail: outputMat is null or empty, returning");
                return;
            }

            // Dispose old thumbnail
            if (thumbnail != null && !thumbnail.isDisposed()) {
                thumbnail.dispose();
            }

            // Create thumbnail
            Mat resized = new Mat();
            double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / outputMat.width(),
                                    (double) PROCESSING_NODE_THUMB_HEIGHT / outputMat.height());
            Imgproc.resize(outputMat, resized,
                new Size(outputMat.width() * scale, outputMat.height() * scale));

            // Convert to SWT Image
            Mat rgb = new Mat();
            if (resized.channels() == 3) {
                Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
            } else if (resized.channels() == 1) {
                Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
            } else {
                rgb = resized;
            }

            int w = rgb.width();
            int h = rgb.height();
            byte[] data = new byte[w * h * 3];
            rgb.get(0, 0, data);

            // Create ImageData with proper scanline padding
            PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
            ImageData imageData = new ImageData(w, h, 24, palette);

            // Copy data row by row to handle scanline padding
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int srcIdx = (y * w + x) * 3;
                    int r = data[srcIdx] & 0xFF;
                    int g = data[srcIdx + 1] & 0xFF;
                    int b = data[srcIdx + 2] & 0xFF;
                    imageData.setPixel(x, y, (r << 16) | (g << 8) | b);
                }
            }

            thumbnail = new Image(display, imageData);
            System.out.println("DEBUG updateThumbnail: Created thumbnail " + thumbnail.getBounds().width + "x" + thumbnail.getBounds().height);
        }

        protected void drawThumbnail(GC gc, int thumbX, int thumbY) {
            if (thumbnail != null && !thumbnail.isDisposed()) {
                gc.drawImage(thumbnail, thumbX, thumbY);
            }
        }

        public void disposeThumbnail() {
            if (thumbnail != null && !thumbnail.isDisposed()) {
                thumbnail.dispose();
            }
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

        // Video support
        private VideoCapture videoCapture = null;
        private boolean isVideo = false;
        private boolean loopVideo = true;
        private double fps = 30.0;

        // FPS mode selection
        // 0 = "Just Once" (0 fps, single shot)
        // 1 = "Automatic" (video fps or 1 fps for static)
        // 2-N = specific fps values (e.g., 1, 5, 10, 15, 24, 30, 60)
        private int fpsMode = 1;  // Default to Automatic
        private static final String[] FPS_OPTIONS = {
            "Just Once", "Automatic", "1 fps", "5 fps", "10 fps", "15 fps", "24 fps", "30 fps", "60 fps"
        };
        private static final double[] FPS_VALUES = {
            0.0, -1.0, 1.0, 5.0, 10.0, 15.0, 24.0, 30.0, 60.0
        };  // -1 means automatic

        // Static image repeat (default 1 fps)
        private boolean repeatImage = true;
        private double staticFps = 1.0;

        public ImageSourceNode(Shell shell, Display display, Canvas canvas, int x, int y) {
            this.shell = shell;
            this.display = display;
            this.parentCanvas = canvas;
            this.x = x;
            this.y = y;
            this.height = SOURCE_NODE_HEIGHT;

            createOverlay();
        }

        private void createOverlay() {
            overlayComposite = new Composite(parentCanvas, SWT.NONE);
            overlayComposite.setLayout(new GridLayout(1, false));
            // Tighten bounds: start at y+22, use exact thumbnail height + small padding
            overlayComposite.setBounds(x + 5, y + 22, width - 10, SOURCE_NODE_THUMB_HEIGHT + 6);

            // Thumbnail label only - Choose button moved to Properties dialog
            Label thumbnailLabel = new Label(overlayComposite, SWT.BORDER | SWT.CENTER);
            GridData gd = new GridData(SWT.FILL, SWT.FILL, true, true);
            gd.heightHint = SOURCE_NODE_THUMB_HEIGHT;
            thumbnailLabel.setLayoutData(gd);
            thumbnailLabel.setText("No image");

            // Add mouse listeners for dragging from thumbnail area
            MouseListener dragMouseListener = new MouseAdapter() {
                @Override
                public void mouseDown(MouseEvent e) {
                    if (e.button == 1) {  // Left click
                        // Convert to canvas coordinates and forward to canvas
                        Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
                        canvasPoint = parentCanvas.toControl(canvasPoint);

                        // Create a synthetic mouse event for the canvas
                        Event event = new Event();
                        event.x = canvasPoint.x;
                        event.y = canvasPoint.y;
                        event.button = e.button;
                        event.stateMask = e.stateMask;
                        parentCanvas.notifyListeners(SWT.MouseDown, event);
                    }
                }

                @Override
                public void mouseUp(MouseEvent e) {
                    // Convert to canvas coordinates and forward to canvas
                    Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
                    canvasPoint = parentCanvas.toControl(canvasPoint);

                    Event event = new Event();
                    event.x = canvasPoint.x;
                    event.y = canvasPoint.y;
                    event.button = e.button;
                    event.stateMask = e.stateMask;
                    parentCanvas.notifyListeners(SWT.MouseUp, event);
                }

                @Override
                public void mouseDoubleClick(MouseEvent e) {
                    showPropertiesDialog();
                }
            };

            // Add mouse move listener for dragging
            MouseMoveListener dragMoveListener = e -> {
                // Convert to canvas coordinates and forward to canvas
                Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
                canvasPoint = parentCanvas.toControl(canvasPoint);

                Event event = new Event();
                event.x = canvasPoint.x;
                event.y = canvasPoint.y;
                event.stateMask = e.stateMask;
                parentCanvas.notifyListeners(SWT.MouseMove, event);
            };

            // Apply listeners to both overlay and thumbnail
            overlayComposite.addMouseListener(dragMouseListener);
            overlayComposite.addMouseMoveListener(dragMoveListener);
            thumbnailLabel.addMouseListener(dragMouseListener);
            thumbnailLabel.addMouseMoveListener(dragMoveListener);

            // Add right-click menu to overlay composite and thumbnail
            Menu contextMenu = new Menu(overlayComposite);
            MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
            editItem.setText("Edit Properties...");
            editItem.addListener(SWT.Selection, evt -> showPropertiesDialog());

            overlayComposite.setMenu(contextMenu);
            thumbnailLabel.setMenu(contextMenu);

            // Ensure the overlay is visible
            overlayComposite.moveAbove(null);
            overlayComposite.layout();
        }

        private void chooseImage() {
            FileDialog dialog = new FileDialog(shell, SWT.OPEN);
            dialog.setText("Select Image or Video");
            dialog.setFilterExtensions(new String[]{
                "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.mp4;*.avi;*.mov;*.mkv;*.webm",
                "*.png;*.jpg;*.jpeg;*.bmp;*.tiff",
                "*.mp4;*.avi;*.mov;*.mkv;*.webm",
                "*.*"
            });
            dialog.setFilterNames(new String[]{
                "All Media Files",
                "Image Files",
                "Video Files",
                "All Files"
            });

            String path = dialog.open();
            if (path != null) {
                imagePath = path;
                loadMedia(path);
            }
        }

        private void loadMedia(String path) {
            // Check if it's a video file
            String lower = path.toLowerCase();
            if (lower.endsWith(".mp4") || lower.endsWith(".avi") ||
                lower.endsWith(".mov") || lower.endsWith(".mkv") ||
                lower.endsWith(".webm")) {
                loadVideo(path);
            } else {
                loadImage(path);
            }
        }

        private void loadVideo(String path) {
            // Release any existing video capture
            if (videoCapture != null) {
                videoCapture.release();
            }

            videoCapture = new VideoCapture(path);
            if (!videoCapture.isOpened()) {
                isVideo = false;
                return;
            }

            isVideo = true;
            fps = videoCapture.get(Videoio.CAP_PROP_FPS);
            if (fps <= 0) fps = 30.0;

            // Read first frame for thumbnail
            Mat firstFrame = new Mat();
            if (videoCapture.read(firstFrame) && !firstFrame.empty()) {
                loadedImage = firstFrame.clone();

                // Create thumbnail
                Mat resized = new Mat();
                double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / firstFrame.width(),
                                        (double) SOURCE_NODE_THUMB_HEIGHT / firstFrame.height());
                Imgproc.resize(firstFrame, resized,
                    new Size(firstFrame.width() * scale, firstFrame.height() * scale));

                if (thumbnail != null) {
                    thumbnail.dispose();
                }
                thumbnail = matToSwtImage(resized);

                // Update the label (thumbnail is now first child)
                Control[] children = overlayComposite.getChildren();
                if (children.length > 0 && children[0] instanceof Label) {
                    Label label = (Label) children[0];
                    label.setText("");
                    label.setImage(thumbnail);
                }

                firstFrame.release();
            }

            // Reset to beginning
            videoCapture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
        }

        public Mat getNextFrame() {
            if (isVideo && videoCapture != null && videoCapture.isOpened()) {
                Mat frame = new Mat();
                if (videoCapture.read(frame)) {
                    return frame;
                } else if (loopVideo) {
                    // Loop back to start
                    videoCapture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
                    if (videoCapture.read(frame)) {
                        return frame;
                    }
                }
                frame.release();
                return null;
            } else if (loadedImage != null && !loadedImage.empty()) {
                return loadedImage.clone();
            }
            return null;
        }

        public boolean isVideoSource() {
            return isVideo;
        }

        public double getFps() {
            // Handle FPS based on mode selection
            double selectedFps = FPS_VALUES[fpsMode];
            if (selectedFps == -1.0) {
                // Automatic mode: use video fps or 1 fps for static
                return isVideo ? fps : 1.0;
            }
            return selectedFps;
        }

        public int getFpsMode() {
            return fpsMode;
        }

        public void setFpsMode(int mode) {
            this.fpsMode = Math.max(0, Math.min(mode, FPS_OPTIONS.length - 1));
        }

        private void loadImage(String path) {
            System.out.println("DEBUG loadImage: Loading image from " + path);
            loadedImage = Imgcodecs.imread(path);

            if (loadedImage.empty()) {
                System.out.println("DEBUG loadImage: Image is empty! File may not exist: " + path);
                return;
            }
            System.out.println("DEBUG loadImage: Loaded image " + loadedImage.width() + "x" + loadedImage.height());

            // Create thumbnail
            Mat resized = new Mat();
            double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / loadedImage.width(),
                                    (double) SOURCE_NODE_THUMB_HEIGHT / loadedImage.height());
            Imgproc.resize(loadedImage, resized,
                new Size(loadedImage.width() * scale, loadedImage.height() * scale));

            // Store the thumbnail Mat for caching
            thumbnailMat = resized;

            if (thumbnail != null) {
                thumbnail.dispose();
            }
            thumbnail = matToSwtImage(resized);

            // Capture the thumbnail reference for the asyncExec closure
            final Image thumbToSet = thumbnail;

            // Update the label - use asyncExec to defer until after UI is fully initialized
            display.asyncExec(() -> {
                if (overlayComposite.isDisposed()) {
                    return;
                }
                if (thumbToSet == null || thumbToSet.isDisposed()) {
                    System.out.println("DEBUG loadImage: ERROR - thumbnail is null or disposed in asyncExec!");
                    return;
                }
                Control[] children = overlayComposite.getChildren();
                System.out.println("DEBUG loadImage: overlayComposite has " + children.length + " children");
                if (children.length > 0 && children[0] instanceof Label) {
                    Label label = (Label) children[0];
                    label.setText("");
                    label.setImage(thumbToSet);

                    // Force pack to resize label for the image
                    label.pack();

                    // Force complete layout refresh
                    overlayComposite.layout(true, true);

                    // Make sure it's visible and on top
                    overlayComposite.setVisible(true);
                    overlayComposite.moveAbove(null);

                    // Force full repaint
                    label.redraw();
                    label.update();
                    overlayComposite.redraw();
                    overlayComposite.update();
                    if (parentCanvas != null && !parentCanvas.isDisposed()) {
                        parentCanvas.redraw();
                        parentCanvas.update();
                    }

                    System.out.println("DEBUG loadImage: Set thumbnail on label, bounds=" + thumbToSet.getBounds() + ", label bounds=" + label.getBounds() + ", composite bounds=" + overlayComposite.getBounds() + ", visible=" + overlayComposite.isVisible());
                } else {
                    System.out.println("DEBUG loadImage: Could not find label at index 1");
                }
            });
        }

        // Thumbnail caching support
        private Mat thumbnailMat = null;

        public void saveThumbnailToCache(String cacheDir) {
            if (thumbnailMat != null && imagePath != null) {
                try {
                    File cacheFolder = new File(cacheDir);
                    if (!cacheFolder.exists()) {
                        cacheFolder.mkdirs();
                    }
                    String thumbPath = getThumbnailCachePath(cacheDir);
                    Imgcodecs.imwrite(thumbPath, thumbnailMat);
                } catch (Exception e) {
                    System.err.println("Failed to save thumbnail: " + e.getMessage());
                }
            }
        }

        public boolean loadThumbnailFromCache(String cacheDir) {
            if (imagePath == null) return false;

            String thumbPath = getThumbnailCachePath(cacheDir);
            File thumbFile = new File(thumbPath);
            if (!thumbFile.exists()) return false;

            try {
                Mat cached = Imgcodecs.imread(thumbPath);
                if (cached.empty()) return false;

                thumbnailMat = cached;
                if (thumbnail != null) {
                    thumbnail.dispose();
                }
                thumbnail = matToSwtImage(cached);

                // Update the label (thumbnail is now first child)
                Control[] children = overlayComposite.getChildren();
                if (children.length > 0 && children[0] instanceof Label) {
                    Label label = (Label) children[0];
                    label.setText("");
                    label.setImage(thumbnail);
                }
                return true;
            } catch (Exception e) {
                return false;
            }
        }

        private String getThumbnailCachePath(String cacheDir) {
            // Create a simple hash from the image path
            int hash = imagePath.hashCode();
            String ext = imagePath.toLowerCase().endsWith(".png") ? ".png" : ".jpg";
            return cacheDir + File.separator + "thumb_" + Math.abs(hash) + ext;
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

            int w = rgb.width();
            int h = rgb.height();
            byte[] data = new byte[w * h * 3];
            rgb.get(0, 0, data);

            // Create ImageData with proper scanline padding
            PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
            ImageData imageData = new ImageData(w, h, 24, palette);

            // Copy data row by row to handle scanline padding
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int srcIdx = (y * w + x) * 3;
                    int r = data[srcIdx] & 0xFF;
                    int g = data[srcIdx + 1] & 0xFF;
                    int b = data[srcIdx + 2] & 0xFF;
                    imageData.setPixel(x, y, (r << 16) | (g << 8) | b);
                }
            }

            return new Image(display, imageData);
        }

        @Override
        public void paint(GC gc) {
            // Update overlay position (must match createOverlay)
            overlayComposite.setBounds(x + 5, y + 22, width - 10, SOURCE_NODE_THUMB_HEIGHT + 6);

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
            gc.drawString("Image Source", x + 10, y + 4, true);
            boldFont.dispose();

            // Draw connection points (output only - this is a source node)
            drawConnectionPoints(gc);
        }

        @Override
        protected void drawConnectionPoints(GC gc) {
            // ImageSourceNode only has output point (it's a source)
            int radius = 6;
            Point output = getOutputPoint();
            gc.setBackground(new Color(255, 230, 200)); // Light orange fill
            gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
            gc.setForeground(new Color(200, 120, 50));  // Orange border
            gc.setLineWidth(2);
            gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
            gc.setLineWidth(1);  // Reset line width
        }

        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Image Source Properties");
            dialog.setLayout(new GridLayout(2, false));
            dialog.setSize(500, 180);

            // Image/Video source row
            Label sourceLabel = new Label(dialog, SWT.NONE);
            sourceLabel.setText("Source:");

            // Source button and display
            Composite sourceComposite = new Composite(dialog, SWT.NONE);
            sourceComposite.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
            sourceComposite.setLayout(new GridLayout(2, false));

            // Editable text field for path (allows copy/paste)
            Text pathText = new Text(sourceComposite, SWT.BORDER);
            String displayPath = imagePath != null ? imagePath : "";
            pathText.setText(displayPath);
            pathText.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

            // Choose button
            Button chooseButton = new Button(sourceComposite, SWT.PUSH);
            chooseButton.setText("Choose...");
            chooseButton.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    chooseImage();
                    // Update the path text field
                    String newPath = imagePath != null ? imagePath : "";
                    pathText.setText(newPath);
                }
            });

            // FPS Mode label
            Label fpsLabel = new Label(dialog, SWT.NONE);
            fpsLabel.setText("FPS Mode:");

            // FPS Mode dropdown
            Combo fpsCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            fpsCombo.setItems(FPS_OPTIONS);
            fpsCombo.select(fpsMode);
            fpsCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

            // OK button
            Button okButton = new Button(dialog, SWT.PUSH);
            okButton.setText("OK");
            okButton.setLayoutData(new GridData(SWT.RIGHT, SWT.CENTER, false, false));
            okButton.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    // Get the path from the text field
                    String newPath = pathText.getText().trim();
                    if (!newPath.isEmpty() && (imagePath == null || !newPath.equals(imagePath))) {
                        // Path changed, load the new image
                        imagePath = newPath;
                        loadImage(newPath);
                    }
                    fpsMode = fpsCombo.getSelectionIndex();
                    dialog.close();
                }
            });

            // Cancel button
            Button cancelButton = new Button(dialog, SWT.PUSH);
            cancelButton.setText("Cancel");
            cancelButton.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    dialog.close();
                }
            });

            dialog.setDefaultButton(okButton);

            // Center dialog on parent
            Rectangle parentBounds = shell.getBounds();
            Rectangle dialogBounds = dialog.getBounds();
            dialog.setLocation(
                parentBounds.x + (parentBounds.width - dialogBounds.width) / 2,
                parentBounds.y + (parentBounds.height - dialogBounds.height) / 2
            );

            dialog.open();
        }

        public Mat getLoadedImage() {
            return loadedImage;
        }
    }

    // Processing node
    // Base class for processing nodes with properties dialog support
    abstract static class ProcessingNode extends PipelineNode {
        protected String name;
        protected Shell shell;
        protected boolean enabled = true;
        protected Runnable onChanged;  // Callback when properties change

        public ProcessingNode(Display display, Shell shell, String name, int x, int y) {
            this.display = display;
            this.shell = shell;
            this.name = name;
            this.x = x;
            this.y = y;
        }

        public void setOnChanged(Runnable onChanged) {
            this.onChanged = onChanged;
        }

        protected void notifyChanged() {
            if (onChanged != null) {
                onChanged.run();
            }
        }

        // Process input Mat and return output Mat
        public abstract Mat process(Mat input);

        // Show properties dialog
        public abstract void showPropertiesDialog();

        // Get description for tooltip
        public abstract String getDescription();

        @Override
        public void paint(GC gc) {
            System.out.println("DEBUG ProcessingNode.paint: " + name + ", thumbnail=" + (thumbnail != null ? (thumbnail.isDisposed() ? "disposed" : thumbnail.getBounds().width + "x" + thumbnail.getBounds().height) : "null"));
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
            gc.drawString(name, x + 10, y + 5, true);
            boldFont.dispose();

            // Draw thumbnail if available
            if (thumbnail != null && !thumbnail.isDisposed()) {
                Rectangle bounds = thumbnail.getBounds();
                int thumbX = x + (width - bounds.width) / 2;
                int thumbY = y + 25;
                gc.drawImage(thumbnail, thumbX, thumbY);
                System.out.println("DEBUG ProcessingNode.paint: Drew thumbnail at " + thumbX + "," + thumbY);
            } else {
                // Draw placeholder
                gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.drawString("(no output)", x + 10, y + 40, true);
                System.out.println("DEBUG ProcessingNode.paint: No thumbnail, drawing placeholder");
            }

            // Draw connection points
            drawConnectionPoints(gc);
        }

        public String getName() {
            return name;
        }

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }
    }

    // Gaussian Blur effect node
    static class GaussianBlurNode extends ProcessingNode {
        private int kernelSizeX = 7;
        private int kernelSizeY = 7;
        private double sigmaX = 0.0;

        public GaussianBlurNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Gaussian Blur", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            // Ensure odd kernel sizes
            int kx = (kernelSizeX % 2 == 0) ? kernelSizeX + 1 : kernelSizeX;
            int ky = (kernelSizeY % 2 == 0) ? kernelSizeY + 1 : kernelSizeY;

            Mat output = new Mat();
            Imgproc.GaussianBlur(input, output, new Size(kx, ky), sigmaX);
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.GaussianBlur(src, ksize, sigmaX)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Gaussian Blur Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Method signature
            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 3;
            sigLabel.setLayoutData(sigGd);

            // Separator
            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 3;
            sep.setLayoutData(sepGd);

            // Kernel Size X
            new Label(dialog, SWT.NONE).setText("Kernel Size X:");
            Scale kxScale = new Scale(dialog, SWT.HORIZONTAL);
            kxScale.setMinimum(1);
            kxScale.setMaximum(31);
            kxScale.setSelection(kernelSizeX);
            kxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kxLabel = new Label(dialog, SWT.NONE);
            kxLabel.setText(String.valueOf(kernelSizeX));
            kxScale.addListener(SWT.Selection, e -> kxLabel.setText(String.valueOf(kxScale.getSelection())));

            // Kernel Size Y
            new Label(dialog, SWT.NONE).setText("Kernel Size Y:");
            Scale kyScale = new Scale(dialog, SWT.HORIZONTAL);
            kyScale.setMinimum(1);
            kyScale.setMaximum(31);
            kyScale.setSelection(kernelSizeY);
            kyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kyLabel = new Label(dialog, SWT.NONE);
            kyLabel.setText(String.valueOf(kernelSizeY));
            kyScale.addListener(SWT.Selection, e -> kyLabel.setText(String.valueOf(kyScale.getSelection())));

            // Sigma X
            new Label(dialog, SWT.NONE).setText("Sigma X:");
            Scale sigmaScale = new Scale(dialog, SWT.HORIZONTAL);
            sigmaScale.setMinimum(0);
            sigmaScale.setMaximum(100);
            sigmaScale.setSelection((int)(sigmaX * 10));
            sigmaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label sigmaLabel = new Label(dialog, SWT.NONE);
            sigmaLabel.setText(sigmaX == 0 ? "0 (auto)" : String.format("%.1f", sigmaX));
            sigmaScale.addListener(SWT.Selection, e -> {
                double val = sigmaScale.getSelection() / 10.0;
                sigmaLabel.setText(val == 0 ? "0 (auto)" : String.format("%.1f", val));
            });

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSizeX = kxScale.getSelection();
                kernelSizeY = kyScale.getSelection();
                sigmaX = sigmaScale.getSelection() / 10.0;
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            // Position dialog near cursor
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Grayscale / Color Conversion node
    static class GrayscaleNode extends ProcessingNode {
        private static final String[] CONVERSION_NAMES = {
            "BGR to Grayscale", "BGR to RGB", "BGR to HSV", "BGR to HLS",
            "BGR to LAB", "BGR to LUV", "BGR to YCrCb", "BGR to XYZ"
        };
        private static final int[] CONVERSION_CODES = {
            Imgproc.COLOR_BGR2GRAY, Imgproc.COLOR_BGR2RGB, Imgproc.COLOR_BGR2HSV,
            Imgproc.COLOR_BGR2HLS, Imgproc.COLOR_BGR2Lab, Imgproc.COLOR_BGR2Luv,
            Imgproc.COLOR_BGR2YCrCb, Imgproc.COLOR_BGR2XYZ
        };
        private int conversionIndex = 0;

        public GrayscaleNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Color Convert", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            Mat output = new Mat();
            Imgproc.cvtColor(input, output, CONVERSION_CODES[conversionIndex]);

            // Convert grayscale back to BGR for display
            if (output.channels() == 1) {
                Mat bgr = new Mat();
                Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                output.release();
                output = bgr;
            }
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.cvtColor(src, code)";
        }

        @Override
        public void showPropertiesDialog() {
            System.out.println("DEBUG showPropertiesDialog: conversionIndex=" + conversionIndex + " for this=" + System.identityHashCode(this));
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Color Conversion Properties");
            dialog.setLayout(new GridLayout(2, false));

            // Method signature
            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 2;
            sigLabel.setLayoutData(sigGd);

            // Separator
            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 2;
            sep.setLayoutData(sepGd);

            new Label(dialog, SWT.NONE).setText("Conversion:");
            Combo combo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            combo.setItems(CONVERSION_NAMES);
            combo.select(conversionIndex);

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 2;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                conversionIndex = combo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            // Position dialog near cursor
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Invert effect node
    static class InvertNode extends ProcessingNode {
        public InvertNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Invert", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            Mat output = new Mat();
            org.opencv.core.Core.bitwise_not(input, output);
            return output;
        }

        @Override
        public String getDescription() {
            return "255 - pixel value";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Invert Properties");
            dialog.setLayout(new GridLayout(1, false));

            // Method signature
            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));

            // Separator
            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            sep.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

            new Label(dialog, SWT.NONE).setText("Inverts all pixel values (negative image).\nNo parameters to configure.");

            Button okBtn = new Button(dialog, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, true, false));
            okBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            // Position dialog near cursor
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Threshold effect node
    static class ThresholdNode extends ProcessingNode {
        private static final String[] TYPE_NAMES = {
            "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"
        };
        private static final int[] TYPE_CODES = {
            Imgproc.THRESH_BINARY, Imgproc.THRESH_BINARY_INV, Imgproc.THRESH_TRUNC,
            Imgproc.THRESH_TOZERO, Imgproc.THRESH_TOZERO_INV
        };
        private static final String[] MODIFIER_NAMES = {"None", "OTSU", "TRIANGLE"};
        private static final int[] MODIFIER_CODES = {0, Imgproc.THRESH_OTSU, Imgproc.THRESH_TRIANGLE};

        private int threshValue = 127;
        private int maxValue = 255;
        private int typeIndex = 0;
        private int modifierIndex = 0;

        public ThresholdNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Threshold", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            int combinedType = TYPE_CODES[typeIndex] | MODIFIER_CODES[modifierIndex];
            Mat output = new Mat();

            // OTSU and TRIANGLE require grayscale
            if (modifierIndex > 0) {
                Mat gray = new Mat();
                if (input.channels() == 3) {
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                } else {
                    gray = input.clone();
                }
                Imgproc.threshold(gray, output, threshValue, maxValue, combinedType);
                gray.release();

                // Convert back to BGR
                Mat bgr = new Mat();
                Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                output.release();
                output = bgr;
            } else {
                Imgproc.threshold(input, output, threshValue, maxValue, combinedType);
            }
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.threshold(src, thresh, maxval, type)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Threshold Properties");
            dialog.setLayout(new GridLayout(2, false));

            // Method signature
            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 2;
            sigLabel.setLayoutData(sigGd);

            // Separator
            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 2;
            sep.setLayoutData(sepGd);

            // Threshold value
            new Label(dialog, SWT.NONE).setText("Threshold:");
            Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
            threshScale.setMinimum(0);
            threshScale.setMaximum(255);
            threshScale.setSelection(threshValue);
            threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label threshLabel = new Label(dialog, SWT.NONE);
            threshLabel.setText(String.valueOf(threshValue));
            threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

            // Max value
            new Label(dialog, SWT.NONE).setText("Max Value:");
            Scale maxScale = new Scale(dialog, SWT.HORIZONTAL);
            maxScale.setMinimum(0);
            maxScale.setMaximum(255);
            maxScale.setSelection(maxValue);
            maxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label maxLabel = new Label(dialog, SWT.NONE);
            maxLabel.setText(String.valueOf(maxValue));
            maxScale.addListener(SWT.Selection, e -> maxLabel.setText(String.valueOf(maxScale.getSelection())));

            // Type
            new Label(dialog, SWT.NONE).setText("Type:");
            Combo typeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            typeCombo.setItems(TYPE_NAMES);
            typeCombo.select(typeIndex);

            // Modifier
            new Label(dialog, SWT.NONE).setText("Modifier:");
            Combo modCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            modCombo.setItems(MODIFIER_NAMES);
            modCombo.select(modifierIndex);

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                threshValue = threshScale.getSelection();
                maxValue = maxScale.getSelection();
                typeIndex = typeCombo.getSelectionIndex();
                modifierIndex = modCombo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            // Position dialog near cursor
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Gain effect node
    static class GainNode extends ProcessingNode {
        private double gain = 1.0;

        public GainNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Gain", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            Mat output = new Mat();
            input.convertTo(output, -1, gain, 0);
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.multiply(src, gain)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Gain Properties");
            dialog.setLayout(new GridLayout(2, false));

            // Method signature
            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 2;
            sigLabel.setLayoutData(sigGd);

            // Separator
            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 2;
            sep.setLayoutData(sepGd);

            new Label(dialog, SWT.NONE).setText("Gain (0.1x - 10x):");
            Scale gainScale = new Scale(dialog, SWT.HORIZONTAL);
            gainScale.setMinimum(1);
            gainScale.setMaximum(100);
            // Use logarithmic mapping: scale value = log10(gain) * 50 + 50
            gainScale.setSelection((int)(Math.log10(gain) * 50 + 50));
            gainScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label gainLabel = new Label(dialog, SWT.NONE);
            gainLabel.setText(String.format("%.2fx", gain));
            gainScale.addListener(SWT.Selection, e -> {
                double logVal = (gainScale.getSelection() - 50) / 50.0;
                double g = Math.pow(10, logVal);
                gainLabel.setText(String.format("%.2fx", g));
            });

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                double logVal = (gainScale.getSelection() - 50) / 50.0;
                gain = Math.pow(10, logVal);
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            // Position dialog near cursor
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Canny Edge Detection node
    static class CannyEdgeNode extends ProcessingNode {
        private static final String[] APERTURE_SIZES = {"3", "5", "7"};
        private static final int[] APERTURE_VALUES = {3, 5, 7};

        private int threshold1 = 30;      // Lower threshold
        private int threshold2 = 150;     // Upper threshold
        private int apertureIndex = 0;    // Index into APERTURE_VALUES (default 3)
        private boolean l2Gradient = false;

        public CannyEdgeNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Canny Edge", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }

            // Convert to grayscale if needed
            Mat gray;
            if (input.channels() == 3) {
                gray = new Mat();
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input;
            }

            // Apply Canny edge detection
            Mat edges = new Mat();
            Imgproc.Canny(gray, edges, threshold1, threshold2,
                         APERTURE_VALUES[apertureIndex], l2Gradient);

            // Convert back to BGR for display
            Mat output = new Mat();
            Imgproc.cvtColor(edges, output, Imgproc.COLOR_GRAY2BGR);

            // Clean up temp mat if we created it
            if (gray != input) {
                gray.release();
            }
            edges.release();

            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.Canny(image, threshold1, threshold2)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Canny Edge Detection Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Method signature
            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 3;
            sigLabel.setLayoutData(sigGd);

            // Separator
            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 3;
            sep.setLayoutData(sepGd);

            // Threshold 1 (lower)
            new Label(dialog, SWT.NONE).setText("Threshold 1 (lower):");
            Scale t1Scale = new Scale(dialog, SWT.HORIZONTAL);
            t1Scale.setMinimum(0);
            t1Scale.setMaximum(500);
            t1Scale.setSelection(threshold1);
            t1Scale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label t1Label = new Label(dialog, SWT.NONE);
            t1Label.setText(String.valueOf(threshold1));
            t1Scale.addListener(SWT.Selection, e -> t1Label.setText(String.valueOf(t1Scale.getSelection())));

            // Threshold 2 (upper)
            new Label(dialog, SWT.NONE).setText("Threshold 2 (upper):");
            Scale t2Scale = new Scale(dialog, SWT.HORIZONTAL);
            t2Scale.setMinimum(0);
            t2Scale.setMaximum(500);
            t2Scale.setSelection(threshold2);
            t2Scale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label t2Label = new Label(dialog, SWT.NONE);
            t2Label.setText(String.valueOf(threshold2));
            t2Scale.addListener(SWT.Selection, e -> t2Label.setText(String.valueOf(t2Scale.getSelection())));

            // Aperture Size (dropdown)
            new Label(dialog, SWT.NONE).setText("Aperture Size:");
            Combo apertureCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            apertureCombo.setItems(APERTURE_SIZES);
            apertureCombo.select(apertureIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            apertureCombo.setLayoutData(comboGd);

            // L2 Gradient (checkbox)
            org.eclipse.swt.widgets.Button l2Check = new org.eclipse.swt.widgets.Button(dialog, SWT.CHECK);
            l2Check.setText("L2 Gradient");
            l2Check.setSelection(l2Gradient);
            GridData checkGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
            checkGd.horizontalSpan = 3;
            l2Check.setLayoutData(checkGd);

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                threshold1 = t1Scale.getSelection();
                threshold2 = t2Scale.getSelection();
                apertureIndex = apertureCombo.getSelectionIndex();
                l2Gradient = l2Check.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            // Position dialog near cursor
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
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

    // Dangling connection with one end free (source fixed, target free)
    static class DanglingConnection {
        PipelineNode source;
        Point freeEnd;

        public DanglingConnection(PipelineNode source, Point freeEnd) {
            this.source = source;
            this.freeEnd = new Point(freeEnd.x, freeEnd.y);
        }
    }

    // Reverse dangling connection (target fixed, source free)
    static class ReverseDanglingConnection {
        PipelineNode target;
        Point freeEnd;

        public ReverseDanglingConnection(PipelineNode target, Point freeEnd) {
            this.target = target;
            this.freeEnd = new Point(freeEnd.x, freeEnd.y);
        }
    }

    // Free connection (both ends free, no nodes attached)
    static class FreeConnection {
        Point startEnd;  // Non-arrow end
        Point arrowEnd;  // Arrow end

        public FreeConnection(Point startEnd, Point arrowEnd) {
            this.startEnd = new Point(startEnd.x, startEnd.y);
            this.arrowEnd = new Point(arrowEnd.x, arrowEnd.y);
        }
    }
}
