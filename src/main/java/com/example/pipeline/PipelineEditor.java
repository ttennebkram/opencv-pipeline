package com.example.pipeline;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;
import org.eclipse.swt.custom.SashForm;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.CLAHE;
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
    private Set<DanglingConnection> selectedDanglingConnections = new HashSet<>();
    private Set<ReverseDanglingConnection> selectedReverseDanglingConnections = new HashSet<>();
    private Set<FreeConnection> selectedFreeConnections = new HashSet<>();
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
        // Create scrollable container for the toolbar
        org.eclipse.swt.custom.ScrolledComposite scrolledToolbar = new org.eclipse.swt.custom.ScrolledComposite(shell, SWT.V_SCROLL | SWT.BORDER);
        scrolledToolbar.setLayoutData(new GridData(SWT.FILL, SWT.FILL, false, true));
        scrolledToolbar.setExpandHorizontal(true);
        scrolledToolbar.setExpandVertical(true);

        Composite toolbar = new Composite(scrolledToolbar, SWT.NONE);
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

        // Basic Effects
        Label basicLabel = new Label(toolbar, SWT.NONE);
        basicLabel.setText("Basic:");
        basicLabel.setFont(boldFont);

        createNodeButton(toolbar, "Grayscale", () -> addEffectNode("Grayscale"));
        createNodeButton(toolbar, "Invert", () -> addEffectNode("Invert"));
        createNodeButton(toolbar, "Gain", () -> addEffectNode("Gain"));
        createNodeButton(toolbar, "Threshold", () -> addEffectNode("Threshold"));
        createNodeButton(toolbar, "Adaptive Threshold", () -> addEffectNode("AdaptiveThreshold"));
        createNodeButton(toolbar, "CLAHE", () -> addEffectNode("CLAHE"));
        createNodeButton(toolbar, "Color In Range", () -> addEffectNode("ColorInRange"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Blur Effects
        Label blurLabel = new Label(toolbar, SWT.NONE);
        blurLabel.setText("Blur:");
        blurLabel.setFont(boldFont);

        createNodeButton(toolbar, "Gaussian Blur", () -> addEffectNode("GaussianBlur"));
        createNodeButton(toolbar, "Median Blur", () -> addEffectNode("MedianBlur"));
        createNodeButton(toolbar, "Bilateral Filter", () -> addEffectNode("BilateralFilter"));
        createNodeButton(toolbar, "Mean Shift", () -> addEffectNode("MeanShift"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Edge Detection
        Label edgeLabel = new Label(toolbar, SWT.NONE);
        edgeLabel.setText("Edge Detection:");
        edgeLabel.setFont(boldFont);

        createNodeButton(toolbar, "Canny Edges", () -> addEffectNode("CannyEdge"));
        createNodeButton(toolbar, "Laplacian Edges", () -> addEffectNode("Laplacian"));
        createNodeButton(toolbar, "Sobel Edges", () -> addEffectNode("Sobel"));
        createNodeButton(toolbar, "Scharr Edges", () -> addEffectNode("Scharr"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Morphological
        Label morphLabel = new Label(toolbar, SWT.NONE);
        morphLabel.setText("Morphological:");
        morphLabel.setFont(boldFont);

        createNodeButton(toolbar, "Erode", () -> addEffectNode("Erode"));
        createNodeButton(toolbar, "Dilate", () -> addEffectNode("Dilate"));
        createNodeButton(toolbar, "Morph Open", () -> addEffectNode("MorphOpen"));
        createNodeButton(toolbar, "Morph Close", () -> addEffectNode("MorphClose"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Detection
        Label detectLabel = new Label(toolbar, SWT.NONE);
        detectLabel.setText("Detection:");
        detectLabel.setFont(boldFont);

        createNodeButton(toolbar, "Hough Circles", () -> addEffectNode("HoughCircles"));
        createNodeButton(toolbar, "Hough Lines", () -> addEffectNode("HoughLines"));
        createNodeButton(toolbar, "Contours", () -> addEffectNode("Contours"));
        createNodeButton(toolbar, "Harris Corners", () -> addEffectNode("HarrisCorners"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

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

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Recent Files section
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

        // Separator before buttons
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

        // Set the toolbar as content of scrolled composite
        scrolledToolbar.setContent(toolbar);
        scrolledToolbar.setMinSize(toolbar.computeSize(SWT.DEFAULT, SWT.DEFAULT));
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

            // Write to file
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter writer = new FileWriter(path)) {
                gson.toJson(root, writer);
            }


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
                            node.imagePath = imgPath;
                            // Load the media (creates thumbnail and loads image for execution)
                            node.loadMedia(imgPath);
                        } else {
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
                                } else {
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
                canvas.setFocus();  // Ensure canvas has focus for keyboard events
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

        // Keyboard shortcuts - use Display filter to catch key events reliably on macOS
        display.addFilter(SWT.KeyDown, event -> {
            // Only handle keys when our shell is active
            if (shell.isDisposed() || !shell.isVisible()) return;

            // Cmd-A to select all nodes
            if (event.character == 'a' && (event.stateMask & SWT.MOD1) != 0) {
                selectedNodes.addAll(nodes);
                canvas.redraw();
                event.doit = false; // Consume the event
            }
            // Delete or Backspace to delete selected nodes and connections
            // SWT.DEL = 127 (forward delete), SWT.BS = 8 (backspace)
            else if (event.keyCode == SWT.DEL || event.keyCode == SWT.BS ||
                     event.character == '\b' || event.character == 127) {
                boolean hasSelection = !selectedNodes.isEmpty() || !selectedConnections.isEmpty() ||
                    !selectedDanglingConnections.isEmpty() || !selectedReverseDanglingConnections.isEmpty() ||
                    !selectedFreeConnections.isEmpty();

                // Delete selected connections first
                if (!selectedConnections.isEmpty()) {
                    connections.removeAll(selectedConnections);
                    selectedConnections.clear();
                }

                // Delete selected dangling connections
                if (!selectedDanglingConnections.isEmpty()) {
                    danglingConnections.removeAll(selectedDanglingConnections);
                    selectedDanglingConnections.clear();
                }

                // Delete selected reverse dangling connections
                if (!selectedReverseDanglingConnections.isEmpty()) {
                    reverseDanglingConnections.removeAll(selectedReverseDanglingConnections);
                    selectedReverseDanglingConnections.clear();
                }

                // Delete selected free connections
                if (!selectedFreeConnections.isEmpty()) {
                    freeConnections.removeAll(selectedFreeConnections);
                    selectedFreeConnections.clear();
                }

                if (!selectedNodes.isEmpty()) {
                    // Convert connections to dangling connections before removing nodes
                    List<Connection> connectionsToRemove = new ArrayList<>();
                    for (Connection conn : connections) {
                        boolean sourceDeleted = selectedNodes.contains(conn.source);
                        boolean targetDeleted = selectedNodes.contains(conn.target);

                        if (sourceDeleted && targetDeleted) {
                            // Both ends deleted - remove the connection entirely
                            connectionsToRemove.add(conn);
                        } else if (sourceDeleted) {
                            // Source deleted - convert to reverse dangling connection
                            // (has target but dangling source)
                            Point sourcePoint = conn.source.getOutputPoint();
                            reverseDanglingConnections.add(new ReverseDanglingConnection(
                                conn.target, sourcePoint));
                            connectionsToRemove.add(conn);
                        } else if (targetDeleted) {
                            // Target deleted - convert to dangling connection
                            // (has source but dangling target)
                            Point targetPoint = conn.target.getInputPoint();
                            danglingConnections.add(new DanglingConnection(
                                conn.source, targetPoint));
                            connectionsToRemove.add(conn);
                        }
                    }
                    connections.removeAll(connectionsToRemove);

                    // Convert dangling connections where the source is deleted to free connections
                    List<DanglingConnection> danglingToRemove = new ArrayList<>();
                    for (DanglingConnection dc : danglingConnections) {
                        if (selectedNodes.contains(dc.source)) {
                            Point sourcePoint = dc.source.getOutputPoint();
                            freeConnections.add(new FreeConnection(sourcePoint, dc.freeEnd));
                            danglingToRemove.add(dc);
                        }
                    }
                    danglingConnections.removeAll(danglingToRemove);

                    // Convert reverse dangling connections where the target is deleted to free connections
                    List<ReverseDanglingConnection> reverseDanglingToRemove = new ArrayList<>();
                    for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
                        if (selectedNodes.contains(rdc.target)) {
                            Point targetPoint = rdc.target.getInputPoint();
                            freeConnections.add(new FreeConnection(rdc.freeEnd, targetPoint));
                            reverseDanglingToRemove.add(rdc);
                        }
                    }
                    reverseDanglingConnections.removeAll(reverseDanglingToRemove);

                    // Remove the nodes
                    nodes.removeAll(selectedNodes);

                    // Clear selection
                    selectedNodes.clear();
                }

                // If we had any selection, redraw and re-execute pipeline
                if (hasSelection) {
                    canvas.redraw();
                    executePipeline();
                    event.doit = false; // Consume the event
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
            return;
        }

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
                // Use the node's process method instead of hardcoded switch
                currentMat = procNode.process(currentMat);
                // Set output on this node for thumbnail
                nextNode.setOutputMat(currentMat);
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

            case "Invert":
                // Invert: 255 - pixel value
                Core.bitwise_not(input, output);
                break;

            case "MedianBlur":
                // Median blur with kernel size 5 (must be odd)
                Imgproc.medianBlur(input, output, 5);
                break;

            case "BilateralFilter":
                // Bilateral filter: edge-preserving smoothing
                // diameter=9, sigmaColor=75, sigmaSpace=75
                Imgproc.bilateralFilter(input, output, 9, 75, 75);
                break;

            case "Laplacian":
                // Laplacian edge detection
                Mat grayForLaplacian = input;
                if (input.channels() == 3) {
                    grayForLaplacian = new Mat();
                    Imgproc.cvtColor(input, grayForLaplacian, Imgproc.COLOR_BGR2GRAY);
                }
                Mat laplacianResult = new Mat();
                Imgproc.Laplacian(grayForLaplacian, laplacianResult, CvType.CV_64F, 3, 1.0, 0);
                // Convert to absolute values and back to 8-bit
                Core.convertScaleAbs(laplacianResult, output);
                break;

            case "Sobel":
                // Sobel edge detection (combined X and Y)
                Mat grayForSobel = input;
                if (input.channels() == 3) {
                    grayForSobel = new Mat();
                    Imgproc.cvtColor(input, grayForSobel, Imgproc.COLOR_BGR2GRAY);
                }
                Mat sobelX = new Mat();
                Mat sobelY = new Mat();
                Imgproc.Sobel(grayForSobel, sobelX, CvType.CV_64F, 1, 0, 3);
                Imgproc.Sobel(grayForSobel, sobelY, CvType.CV_64F, 0, 1, 3);
                Mat absX = new Mat();
                Mat absY = new Mat();
                Core.convertScaleAbs(sobelX, absX);
                Core.convertScaleAbs(sobelY, absY);
                Core.addWeighted(absX, 0.5, absY, 0.5, 0, output);
                break;

            case "Erode":
                // Morphological erosion
                Mat erodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
                Imgproc.erode(input, output, erodeKernel);
                break;

            case "Dilate":
                // Morphological dilation
                Mat dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
                Imgproc.dilate(input, output, dilateKernel);
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
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint();
            Point end = conn.target.getInputPoint();

            // Highlight selected connections
            if (selectedConnections.contains(conn)) {
                gc.setLineWidth(3);
                gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
            } else {
                gc.setLineWidth(2);
                gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
            }

            gc.drawLine(start.x, start.y, end.x, end.y);
            drawArrow(gc, start, end);
        }

        // Draw dangling connections with dashed lines
        gc.setLineStyle(SWT.LINE_DASH);
        for (DanglingConnection dangling : danglingConnections) {
            Point start = dangling.source.getOutputPoint();
            // Highlight selected dangling connections
            if (selectedDanglingConnections.contains(dangling)) {
                gc.setLineWidth(3);
                gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
                gc.setBackground(display.getSystemColor(SWT.COLOR_CYAN));
            } else {
                gc.setLineWidth(2);
                gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
            }
            gc.drawLine(start.x, start.y, dangling.freeEnd.x, dangling.freeEnd.y);
            drawArrow(gc, start, dangling.freeEnd);
            // Draw small circles at both ends to make them easier to grab
            gc.fillOval(start.x - 4, start.y - 4, 8, 8);
            gc.fillOval(dangling.freeEnd.x - 4, dangling.freeEnd.y - 4, 8, 8);
        }
        // Draw reverse dangling connections (target fixed, source free)
        for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
            Point end = dangling.target.getInputPoint();
            // Highlight selected reverse dangling connections
            if (selectedReverseDanglingConnections.contains(dangling)) {
                gc.setLineWidth(3);
                gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
                gc.setBackground(display.getSystemColor(SWT.COLOR_CYAN));
            } else {
                gc.setLineWidth(2);
                gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
            }
            gc.drawLine(dangling.freeEnd.x, dangling.freeEnd.y, end.x, end.y);
            // Draw small circles at both ends to make them easier to grab
            gc.fillOval(dangling.freeEnd.x - 4, dangling.freeEnd.y - 4, 8, 8);
            gc.fillOval(end.x - 4, end.y - 4, 8, 8);
        }
        // Draw free connections (both ends free)
        for (FreeConnection free : freeConnections) {
            // Highlight selected free connections
            if (selectedFreeConnections.contains(free)) {
                gc.setLineWidth(3);
                gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
                gc.setBackground(display.getSystemColor(SWT.COLOR_CYAN));
            } else {
                gc.setLineWidth(2);
                gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
            }
            gc.drawLine(free.startEnd.x, free.startEnd.y, free.arrowEnd.x, free.arrowEnd.y);
            drawArrow(gc, free.startEnd, free.arrowEnd);
            // Draw small circles at both ends to make them easier to grab
            gc.fillOval(free.startEnd.x - 4, free.startEnd.y - 4, 8, 8);
            gc.fillOval(free.arrowEnd.x - 4, free.arrowEnd.y - 4, 8, 8);
        }
        gc.setLineStyle(SWT.LINE_SOLID);
        gc.setLineWidth(1);

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

    // Helper method to calculate distance from point to line segment
    private double pointToLineDistance(Point p, Point lineStart, Point lineEnd) {
        double dx = lineEnd.x - lineStart.x;
        double dy = lineEnd.y - lineStart.y;
        double lengthSquared = dx * dx + dy * dy;

        if (lengthSquared == 0) {
            // Line segment is actually a point
            return Math.sqrt(Math.pow(p.x - lineStart.x, 2) + Math.pow(p.y - lineStart.y, 2));
        }

        // Calculate projection parameter t
        double t = ((p.x - lineStart.x) * dx + (p.y - lineStart.y) * dy) / lengthSquared;
        t = Math.max(0, Math.min(1, t)); // Clamp to [0, 1]

        // Find closest point on line segment
        double closestX = lineStart.x + t * dx;
        double closestY = lineStart.y + t * dy;

        return Math.sqrt(Math.pow(p.x - closestX, 2) + Math.pow(p.y - closestY, 2));
    }

    // Helper method to clear all dangling connection selections
    private void clearDanglingSelections() {
        selectedDanglingConnections.clear();
        selectedReverseDanglingConnections.clear();
        selectedFreeConnections.clear();
    }

    private void handleMouseDown(MouseEvent e) {
        if (e.button == 1) {
            Point clickPoint = new Point(e.x, e.y);
            int radius = 8; // Slightly larger than visual for easier clicking
            boolean cmdHeld = (e.stateMask & SWT.MOD1) != 0;

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
                        connectionSource = connToRemove.source;
                        connectionEndPoint = clickPoint;
                        connections.remove(connToRemove);
                        canvas.redraw();
                        return;
                    }
                    // Check if there's a reverse dangling connection to this input
                    // If so, allow dragging the arrow end to reconnect to a different node
                    ReverseDanglingConnection reverseToYank = null;
                    for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
                        if (dangling.target == node) {
                            reverseToYank = dangling;
                            break;
                        }
                    }
                    if (reverseToYank != null) {
                        reverseDanglingConnections.remove(reverseToYank);
                        // Set up dragging the arrow end while the source end stays fixed
                        freeConnectionFixedEnd = reverseToYank.freeEnd; // The source end stays put
                        draggingFreeConnectionSource = false; // We're dragging the arrow/target end
                        connectionEndPoint = clickPoint;
                        canvas.redraw();
                        return;
                    }
                    // No existing connection - start a new connection from input point (reverse direction)
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
                if (dist <= radius) {
                    // Yank off the source end - this becomes a completely free connector
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
                            clearDanglingSelections();
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

            // Check for connection line selection (clicking on the line itself, not endpoints)
            double clickThreshold = 5.0; // Distance threshold for click detection

            // Check regular connections
            for (Connection conn : connections) {
                Point start = conn.source.getOutputPoint();
                Point end = conn.target.getInputPoint();
                if (pointToLineDistance(clickPoint, start, end) <= clickThreshold) {
                    if (cmdHeld) {
                        if (selectedConnections.contains(conn)) {
                            selectedConnections.remove(conn);
                        } else {
                            selectedConnections.add(conn);
                        }
                    } else {
                        selectedNodes.clear();
                        selectedConnections.clear();
                        clearDanglingSelections();
                        selectedConnections.add(conn);
                    }
                    canvas.redraw();
                    return;
                }
            }

            // Check dangling connections
            for (DanglingConnection dangling : danglingConnections) {
                Point start = dangling.source.getOutputPoint();
                if (pointToLineDistance(clickPoint, start, dangling.freeEnd) <= clickThreshold) {
                    if (cmdHeld) {
                        if (selectedDanglingConnections.contains(dangling)) {
                            selectedDanglingConnections.remove(dangling);
                        } else {
                            selectedDanglingConnections.add(dangling);
                        }
                    } else {
                        selectedNodes.clear();
                        selectedConnections.clear();
                        clearDanglingSelections();
                        selectedDanglingConnections.add(dangling);
                    }
                    canvas.redraw();
                    return;
                }
            }

            // Check reverse dangling connections
            for (ReverseDanglingConnection dangling : reverseDanglingConnections) {
                Point end = dangling.target.getInputPoint();
                if (pointToLineDistance(clickPoint, dangling.freeEnd, end) <= clickThreshold) {
                    if (cmdHeld) {
                        if (selectedReverseDanglingConnections.contains(dangling)) {
                            selectedReverseDanglingConnections.remove(dangling);
                        } else {
                            selectedReverseDanglingConnections.add(dangling);
                        }
                    } else {
                        selectedNodes.clear();
                        selectedConnections.clear();
                        clearDanglingSelections();
                        selectedReverseDanglingConnections.add(dangling);
                    }
                    canvas.redraw();
                    return;
                }
            }

            // Check free connections
            for (FreeConnection free : freeConnections) {
                if (pointToLineDistance(clickPoint, free.startEnd, free.arrowEnd) <= clickThreshold) {
                    if (cmdHeld) {
                        if (selectedFreeConnections.contains(free)) {
                            selectedFreeConnections.remove(free);
                        } else {
                            selectedFreeConnections.add(free);
                        }
                    } else {
                        selectedNodes.clear();
                        selectedConnections.clear();
                        clearDanglingSelections();
                        selectedFreeConnections.add(free);
                    }
                    canvas.redraw();
                    return;
                }
            }

            // Clicked on empty space - start selection box
            if (!cmdHeld) {
                selectedNodes.clear();
                selectedConnections.clear();
                clearDanglingSelections();
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
            } else if (connectionEndPoint != null) {
                // Create a dangling connection
                danglingConnections.add(new DanglingConnection(connectionSource, connectionEndPoint));
            } else {
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
                    danglingConnections.add(new DanglingConnection(sourceNode, freeConnectionFixedEnd));
                } else {
                    // Not connected - create FreeConnection at current position
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
                    reverseDanglingConnections.add(new ReverseDanglingConnection(targetNode, freeConnectionFixedEnd));
                } else {
                    // Not connected - create FreeConnection at current position
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
            case "MedianBlur":
            case "Median Blur":
                return new MedianBlurNode(display, shell, x, y);
            case "BilateralFilter":
            case "Bilateral Filter":
                return new BilateralFilterNode(display, shell, x, y);
            case "Laplacian":
            case "Edges Laplacian":
                return new LaplacianNode(display, shell, x, y);
            case "Sobel":
            case "Gradient Sobel":
                return new SobelNode(display, shell, x, y);
            case "Erode":
            case "Morph Erode":
                return new ErodeNode(display, shell, x, y);
            case "Dilate":
            case "Morph Dilate":
                return new DilateNode(display, shell, x, y);
            case "MorphOpen":
            case "Morph Open":
                return new MorphOpenNode(display, shell, x, y);
            case "MorphClose":
            case "Morph Close":
                return new MorphCloseNode(display, shell, x, y);
            case "Scharr":
            case "Edges Scharr":
                return new ScharrNode(display, shell, x, y);
            case "AdaptiveThreshold":
            case "Adaptive Threshold":
            case "Threshold (Adaptive)":
                return new AdaptiveThresholdNode(display, shell, x, y);
            case "CLAHE":
            case "CLAHE: Contrast Enhancement":
                return new CLAHENode(display, shell, x, y);
            case "ColorInRange":
            case "Color In Range":
                return new ColorInRangeNode(display, shell, x, y);
            case "MeanShift":
            case "Mean Shift":
            case "Mean Shift Filter":
                return new MeanShiftFilterNode(display, shell, x, y);
            case "HoughCircles":
            case "Hough Circles":
                return new HoughCirclesNode(display, shell, x, y);
            case "HoughLines":
            case "Hough Lines":
                return new HoughLinesNode(display, shell, x, y);
            case "Contours":
                return new ContoursNode(display, shell, x, y);
            case "HarrisCorners":
            case "Harris Corners":
                return new HarrisCornersNode(display, shell, x, y);
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
            if (outputMat == null || outputMat.empty()) {
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
            loadedImage = Imgcodecs.imread(path);

            if (loadedImage.empty()) {
                return;
            }

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
                    return;
                }
                Control[] children = overlayComposite.getChildren();
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

                } else {
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
            } else {
                // Draw placeholder
                gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.drawString("(no output)", x + 10, y + 40, true);
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
            GridData typeGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            typeGd.horizontalSpan = 2;
            typeCombo.setLayoutData(typeGd);

            // Modifier
            new Label(dialog, SWT.NONE).setText("Modifier:");
            Combo modCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            modCombo.setItems(MODIFIER_NAMES);
            modCombo.select(modifierIndex);
            GridData modGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            modGd.horizontalSpan = 2;
            modCombo.setLayoutData(modGd);

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

    // Median Blur node
    static class MedianBlurNode extends ProcessingNode {
        private int kernelSize = 5;

        public MedianBlurNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Median Blur", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            // Ensure odd kernel size
            int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;

            Mat output = new Mat();
            Imgproc.medianBlur(input, output, ksize);
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.medianBlur(src, ksize)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Median Blur Properties");
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

            // Kernel Size
            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
            kScale.setMinimum(1);
            kScale.setMaximum(31);
            kScale.setSelection(kernelSize);
            kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kLabel = new Label(dialog, SWT.NONE);
            kLabel.setText(String.valueOf(kernelSize));
            kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSize = kScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Bilateral Filter node
    static class BilateralFilterNode extends ProcessingNode {
        private int diameter = 9;
        private int sigmaColor = 75;
        private int sigmaSpace = 75;

        public BilateralFilterNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Bilateral Filter", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }
            Mat output = new Mat();
            Imgproc.bilateralFilter(input, output, diameter, sigmaColor, sigmaSpace);
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Bilateral Filter Properties");
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

            // Diameter
            new Label(dialog, SWT.NONE).setText("Diameter:");
            Scale dScale = new Scale(dialog, SWT.HORIZONTAL);
            dScale.setMinimum(1);
            dScale.setMaximum(25);
            dScale.setSelection(diameter);
            dScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label dLabel = new Label(dialog, SWT.NONE);
            dLabel.setText(String.valueOf(diameter));
            dScale.addListener(SWT.Selection, e -> dLabel.setText(String.valueOf(dScale.getSelection())));

            // Sigma Color
            new Label(dialog, SWT.NONE).setText("Sigma Color:");
            Scale scScale = new Scale(dialog, SWT.HORIZONTAL);
            scScale.setMinimum(1);
            scScale.setMaximum(200);
            scScale.setSelection(sigmaColor);
            scScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label scLabel = new Label(dialog, SWT.NONE);
            scLabel.setText(String.valueOf(sigmaColor));
            scScale.addListener(SWT.Selection, e -> scLabel.setText(String.valueOf(scScale.getSelection())));

            // Sigma Space
            new Label(dialog, SWT.NONE).setText("Sigma Space:");
            Scale ssScale = new Scale(dialog, SWT.HORIZONTAL);
            ssScale.setMinimum(1);
            ssScale.setMaximum(200);
            ssScale.setSelection(sigmaSpace);
            ssScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label ssLabel = new Label(dialog, SWT.NONE);
            ssLabel.setText(String.valueOf(sigmaSpace));
            ssScale.addListener(SWT.Selection, e -> ssLabel.setText(String.valueOf(ssScale.getSelection())));

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                diameter = dScale.getSelection();
                sigmaColor = scScale.getSelection();
                sigmaSpace = ssScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Laplacian Edge Detection node
    static class LaplacianNode extends ProcessingNode {
        private static final String[] KERNEL_SIZES = {"1", "3", "5", "7"};
        private int kernelSizeIndex = 1; // Default to 3

        public LaplacianNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Laplacian", x, y);
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

            // Get kernel size
            int ksize = Integer.parseInt(KERNEL_SIZES[kernelSizeIndex]);

            // Apply Laplacian
            Mat laplacian = new Mat();
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F, ksize, 1, 0);

            // Convert to absolute and 8-bit
            Mat absLaplacian = new Mat();
            Core.convertScaleAbs(laplacian, absLaplacian);

            // Convert back to BGR for display
            Mat output = new Mat();
            Imgproc.cvtColor(absLaplacian, output, Imgproc.COLOR_GRAY2BGR);

            // Clean up
            if (gray != input) {
                gray.release();
            }
            laplacian.release();
            absLaplacian.release();

            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.Laplacian(src, ddepth)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Laplacian Properties");
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

            // Kernel Size dropdown
            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            ksizeCombo.setItems(KERNEL_SIZES);
            ksizeCombo.select(kernelSizeIndex);

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 2;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSizeIndex = ksizeCombo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Sobel Edge Detection node
    static class SobelNode extends ProcessingNode {
        private static final String[] KERNEL_SIZES = {"1", "3", "5", "7"};
        private int dx = 1;
        private int dy = 0;
        private int kernelSizeIndex = 1; // Default to 3

        public SobelNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Sobel", x, y);
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

            int ksize = Integer.parseInt(KERNEL_SIZES[kernelSizeIndex]);

            // Compute gradients
            Mat gradX = new Mat();
            Mat gradY = new Mat();

            if (dx > 0) {
                Imgproc.Sobel(gray, gradX, CvType.CV_64F, dx, 0, ksize);
            }
            if (dy > 0) {
                Imgproc.Sobel(gray, gradY, CvType.CV_64F, 0, dy, ksize);
            }

            Mat result;
            if (dx > 0 && dy > 0) {
                // Combine both gradients
                Mat absX = new Mat();
                Mat absY = new Mat();
                Core.convertScaleAbs(gradX, absX);
                Core.convertScaleAbs(gradY, absY);
                result = new Mat();
                Core.addWeighted(absX, 0.5, absY, 0.5, 0, result);
                absX.release();
                absY.release();
            } else if (dx > 0) {
                result = new Mat();
                Core.convertScaleAbs(gradX, result);
            } else {
                result = new Mat();
                Core.convertScaleAbs(gradY, result);
            }

            // Convert back to BGR for display
            Mat output = new Mat();
            Imgproc.cvtColor(result, output, Imgproc.COLOR_GRAY2BGR);

            // Clean up
            if (gray != input) {
                gray.release();
            }
            if (!gradX.empty()) gradX.release();
            if (!gradY.empty()) gradY.release();
            result.release();

            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.Sobel(src, ddepth, dx, dy, ksize)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Sobel Properties");
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

            // dx
            new Label(dialog, SWT.NONE).setText("dx (X derivative):");
            Scale dxScale = new Scale(dialog, SWT.HORIZONTAL);
            dxScale.setMinimum(0);
            dxScale.setMaximum(2);
            dxScale.setSelection(dx);
            dxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label dxLabel = new Label(dialog, SWT.NONE);
            dxLabel.setText(String.valueOf(dx));
            dxScale.addListener(SWT.Selection, e -> dxLabel.setText(String.valueOf(dxScale.getSelection())));

            // dy
            new Label(dialog, SWT.NONE).setText("dy (Y derivative):");
            Scale dyScale = new Scale(dialog, SWT.HORIZONTAL);
            dyScale.setMinimum(0);
            dyScale.setMaximum(2);
            dyScale.setSelection(dy);
            dyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label dyLabel = new Label(dialog, SWT.NONE);
            dyLabel.setText(String.valueOf(dy));
            dyScale.addListener(SWT.Selection, e -> dyLabel.setText(String.valueOf(dyScale.getSelection())));

            // Kernel Size
            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            ksizeCombo.setItems(KERNEL_SIZES);
            ksizeCombo.select(kernelSizeIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            ksizeCombo.setLayoutData(comboGd);

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                dx = dxScale.getSelection();
                dy = dyScale.getSelection();
                // Ensure at least one derivative is non-zero
                if (dx == 0 && dy == 0) {
                    dx = 1;
                }
                kernelSizeIndex = ksizeCombo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Erode (Morphological) node
    static class ErodeNode extends ProcessingNode {
        private static final String[] SHAPE_NAMES = {"Rectangle", "Ellipse", "Cross"};
        private static final int[] SHAPE_VALUES = {Imgproc.MORPH_RECT, Imgproc.MORPH_ELLIPSE, Imgproc.MORPH_CROSS};

        private int kernelSize = 5;
        private int shapeIndex = 0;
        private int iterations = 1;

        public ErodeNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Erode", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }

            // Ensure odd kernel size
            int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;

            // Create structuring element
            Mat kernel = Imgproc.getStructuringElement(SHAPE_VALUES[shapeIndex], new Size(ksize, ksize));

            // Apply erosion
            Mat output = new Mat();
            Imgproc.erode(input, output, kernel, new org.opencv.core.Point(-1, -1), iterations);

            kernel.release();
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.erode(src, kernel, iterations)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Erode Properties");
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

            // Kernel Size
            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
            kScale.setMinimum(1);
            kScale.setMaximum(31);
            kScale.setSelection(kernelSize);
            kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kLabel = new Label(dialog, SWT.NONE);
            kLabel.setText(String.valueOf(kernelSize));
            kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

            // Kernel Shape
            new Label(dialog, SWT.NONE).setText("Kernel Shape:");
            Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            shapeCombo.setItems(SHAPE_NAMES);
            shapeCombo.select(shapeIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            shapeCombo.setLayoutData(comboGd);

            // Iterations
            new Label(dialog, SWT.NONE).setText("Iterations:");
            Scale iterScale = new Scale(dialog, SWT.HORIZONTAL);
            iterScale.setMinimum(1);
            iterScale.setMaximum(10);
            iterScale.setSelection(iterations);
            iterScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label iterLabel = new Label(dialog, SWT.NONE);
            iterLabel.setText(String.valueOf(iterations));
            iterScale.addListener(SWT.Selection, e -> iterLabel.setText(String.valueOf(iterScale.getSelection())));

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSize = kScale.getSelection();
                shapeIndex = shapeCombo.getSelectionIndex();
                iterations = iterScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Dilate (Morphological) node
    static class DilateNode extends ProcessingNode {
        private static final String[] SHAPE_NAMES = {"Rectangle", "Ellipse", "Cross"};
        private static final int[] SHAPE_VALUES = {Imgproc.MORPH_RECT, Imgproc.MORPH_ELLIPSE, Imgproc.MORPH_CROSS};

        private int kernelSize = 5;
        private int shapeIndex = 0;
        private int iterations = 1;

        public DilateNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Dilate", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }

            // Ensure odd kernel size
            int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;

            // Create structuring element
            Mat kernel = Imgproc.getStructuringElement(SHAPE_VALUES[shapeIndex], new Size(ksize, ksize));

            // Apply dilation
            Mat output = new Mat();
            Imgproc.dilate(input, output, kernel, new org.opencv.core.Point(-1, -1), iterations);

            kernel.release();
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.dilate(src, kernel, iterations)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Dilate Properties");
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

            // Kernel Size
            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
            kScale.setMinimum(1);
            kScale.setMaximum(31);
            kScale.setSelection(kernelSize);
            kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kLabel = new Label(dialog, SWT.NONE);
            kLabel.setText(String.valueOf(kernelSize));
            kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

            // Kernel Shape
            new Label(dialog, SWT.NONE).setText("Kernel Shape:");
            Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            shapeCombo.setItems(SHAPE_NAMES);
            shapeCombo.select(shapeIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            shapeCombo.setLayoutData(comboGd);

            // Iterations
            new Label(dialog, SWT.NONE).setText("Iterations:");
            Scale iterScale = new Scale(dialog, SWT.HORIZONTAL);
            iterScale.setMinimum(1);
            iterScale.setMaximum(10);
            iterScale.setSelection(iterations);
            iterScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label iterLabel = new Label(dialog, SWT.NONE);
            iterLabel.setText(String.valueOf(iterations));
            iterScale.addListener(SWT.Selection, e -> iterLabel.setText(String.valueOf(iterScale.getSelection())));

            // Buttons
            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSize = kScale.getSelection();
                shapeIndex = shapeCombo.getSelectionIndex();
                iterations = iterScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Morph Open (Morphological) node
    static class MorphOpenNode extends ProcessingNode {
        private static final String[] SHAPE_NAMES = {"Rectangle", "Ellipse", "Cross"};
        private static final int[] SHAPE_VALUES = {Imgproc.MORPH_RECT, Imgproc.MORPH_ELLIPSE, Imgproc.MORPH_CROSS};

        private int kernelSize = 5;
        private int shapeIndex = 0;
        private int iterations = 1;

        public MorphOpenNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Morph Open", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }

            int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;
            Mat kernel = Imgproc.getStructuringElement(SHAPE_VALUES[shapeIndex], new Size(ksize, ksize));

            Mat output = new Mat();
            Imgproc.morphologyEx(input, output, Imgproc.MORPH_OPEN, kernel, new org.opencv.core.Point(-1, -1), iterations);

            kernel.release();
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Morph Open Properties");
            dialog.setLayout(new GridLayout(3, false));

            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 3;
            sigLabel.setLayoutData(sigGd);

            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 3;
            sep.setLayoutData(sepGd);

            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
            kScale.setMinimum(1);
            kScale.setMaximum(31);
            kScale.setSelection(kernelSize);
            kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kLabel = new Label(dialog, SWT.NONE);
            kLabel.setText(String.valueOf(kernelSize));
            kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

            new Label(dialog, SWT.NONE).setText("Kernel Shape:");
            Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            shapeCombo.setItems(SHAPE_NAMES);
            shapeCombo.select(shapeIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            shapeCombo.setLayoutData(comboGd);

            new Label(dialog, SWT.NONE).setText("Iterations:");
            Scale iterScale = new Scale(dialog, SWT.HORIZONTAL);
            iterScale.setMinimum(1);
            iterScale.setMaximum(10);
            iterScale.setSelection(iterations);
            iterScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label iterLabel = new Label(dialog, SWT.NONE);
            iterLabel.setText(String.valueOf(iterations));
            iterScale.addListener(SWT.Selection, e -> iterLabel.setText(String.valueOf(iterScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSize = kScale.getSelection();
                shapeIndex = shapeCombo.getSelectionIndex();
                iterations = iterScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Morph Close (Morphological) node
    static class MorphCloseNode extends ProcessingNode {
        private static final String[] SHAPE_NAMES = {"Rectangle", "Ellipse", "Cross"};
        private static final int[] SHAPE_VALUES = {Imgproc.MORPH_RECT, Imgproc.MORPH_ELLIPSE, Imgproc.MORPH_CROSS};

        private int kernelSize = 5;
        private int shapeIndex = 0;
        private int iterations = 1;

        public MorphCloseNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Morph Close", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }

            int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;
            Mat kernel = Imgproc.getStructuringElement(SHAPE_VALUES[shapeIndex], new Size(ksize, ksize));

            Mat output = new Mat();
            Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, kernel, new org.opencv.core.Point(-1, -1), iterations);

            kernel.release();
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Morph Close Properties");
            dialog.setLayout(new GridLayout(3, false));

            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 3;
            sigLabel.setLayoutData(sigGd);

            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 3;
            sep.setLayoutData(sepGd);

            new Label(dialog, SWT.NONE).setText("Kernel Size:");
            Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
            kScale.setMinimum(1);
            kScale.setMaximum(31);
            kScale.setSelection(kernelSize);
            kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label kLabel = new Label(dialog, SWT.NONE);
            kLabel.setText(String.valueOf(kernelSize));
            kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

            new Label(dialog, SWT.NONE).setText("Kernel Shape:");
            Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            shapeCombo.setItems(SHAPE_NAMES);
            shapeCombo.select(shapeIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            shapeCombo.setLayoutData(comboGd);

            new Label(dialog, SWT.NONE).setText("Iterations:");
            Scale iterScale = new Scale(dialog, SWT.HORIZONTAL);
            iterScale.setMinimum(1);
            iterScale.setMaximum(10);
            iterScale.setSelection(iterations);
            iterScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label iterLabel = new Label(dialog, SWT.NONE);
            iterLabel.setText(String.valueOf(iterations));
            iterScale.addListener(SWT.Selection, e -> iterLabel.setText(String.valueOf(iterScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                kernelSize = kScale.getSelection();
                shapeIndex = shapeCombo.getSelectionIndex();
                iterations = iterScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Scharr Edge Detection node
    static class ScharrNode extends ProcessingNode {
        private static final String[] DIRECTIONS = {"X", "Y", "Both"};
        private int directionIndex = 2; // Default to Both

        public ScharrNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Scharr", x, y);
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

            Mat result;
            if (directionIndex == 0) { // X only
                Mat scharrX = new Mat();
                Imgproc.Scharr(gray, scharrX, CvType.CV_64F, 1, 0);
                result = new Mat();
                Core.convertScaleAbs(scharrX, result);
                scharrX.release();
            } else if (directionIndex == 1) { // Y only
                Mat scharrY = new Mat();
                Imgproc.Scharr(gray, scharrY, CvType.CV_64F, 0, 1);
                result = new Mat();
                Core.convertScaleAbs(scharrY, result);
                scharrY.release();
            } else { // Both
                Mat scharrX = new Mat();
                Mat scharrY = new Mat();
                Imgproc.Scharr(gray, scharrX, CvType.CV_64F, 1, 0);
                Imgproc.Scharr(gray, scharrY, CvType.CV_64F, 0, 1);

                Mat absX = new Mat();
                Mat absY = new Mat();
                Core.convertScaleAbs(scharrX, absX);
                Core.convertScaleAbs(scharrY, absY);

                result = new Mat();
                Core.addWeighted(absX, 0.5, absY, 0.5, 0, result);

                scharrX.release();
                scharrY.release();
                absX.release();
                absY.release();
            }

            // Convert back to BGR for display
            Mat output = new Mat();
            Imgproc.cvtColor(result, output, Imgproc.COLOR_GRAY2BGR);

            if (gray != input) {
                gray.release();
            }
            result.release();

            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.Scharr(src, ddepth, dx, dy)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Scharr Properties");
            dialog.setLayout(new GridLayout(2, false));

            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 2;
            sigLabel.setLayoutData(sigGd);

            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 2;
            sep.setLayoutData(sepGd);

            new Label(dialog, SWT.NONE).setText("Direction:");
            Combo dirCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            dirCombo.setItems(DIRECTIONS);
            dirCombo.select(directionIndex);

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 2;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                directionIndex = dirCombo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Adaptive Threshold node
    static class AdaptiveThresholdNode extends ProcessingNode {
        private static final String[] ADAPTIVE_METHODS = {"Mean", "Gaussian"};
        private static final int[] ADAPTIVE_VALUES = {Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C};
        private static final String[] THRESH_TYPES = {"Binary", "Binary Inv"};
        private static final int[] THRESH_VALUES = {Imgproc.THRESH_BINARY, Imgproc.THRESH_BINARY_INV};

        private int maxValue = 255;
        private int methodIndex = 1; // Gaussian
        private int typeIndex = 0; // Binary
        private int blockSize = 11;
        private int cValue = 2;

        public AdaptiveThresholdNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Adaptive Threshold", x, y);
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

            // Ensure block size is odd
            int bsize = (blockSize % 2 == 0) ? blockSize + 1 : blockSize;

            Mat thresh = new Mat();
            Imgproc.adaptiveThreshold(gray, thresh, maxValue,
                ADAPTIVE_VALUES[methodIndex], THRESH_VALUES[typeIndex], bsize, cValue);

            // Convert back to BGR for display
            Mat output = new Mat();
            Imgproc.cvtColor(thresh, output, Imgproc.COLOR_GRAY2BGR);

            if (gray != input) {
                gray.release();
            }
            thresh.release();

            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.adaptiveThreshold(src, maxValue, method, type, blockSize, C)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Adaptive Threshold Properties");
            dialog.setLayout(new GridLayout(3, false));

            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 3;
            sigLabel.setLayoutData(sigGd);

            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 3;
            sep.setLayoutData(sepGd);

            // Max Value
            new Label(dialog, SWT.NONE).setText("Max Value:");
            Scale maxScale = new Scale(dialog, SWT.HORIZONTAL);
            maxScale.setMinimum(0);
            maxScale.setMaximum(255);
            maxScale.setSelection(maxValue);
            maxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label maxLabel = new Label(dialog, SWT.NONE);
            maxLabel.setText(String.valueOf(maxValue));
            maxScale.addListener(SWT.Selection, e -> maxLabel.setText(String.valueOf(maxScale.getSelection())));

            // Adaptive Method
            new Label(dialog, SWT.NONE).setText("Adaptive Method:");
            Combo methodCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            methodCombo.setItems(ADAPTIVE_METHODS);
            methodCombo.select(methodIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            methodCombo.setLayoutData(comboGd);

            // Threshold Type
            new Label(dialog, SWT.NONE).setText("Threshold Type:");
            Combo typeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            typeCombo.setItems(THRESH_TYPES);
            typeCombo.select(typeIndex);
            GridData typeGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            typeGd.horizontalSpan = 2;
            typeCombo.setLayoutData(typeGd);

            // Block Size
            new Label(dialog, SWT.NONE).setText("Block Size:");
            Scale blockScale = new Scale(dialog, SWT.HORIZONTAL);
            blockScale.setMinimum(3);
            blockScale.setMaximum(99);
            blockScale.setSelection(blockSize);
            blockScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label blockLabel = new Label(dialog, SWT.NONE);
            blockLabel.setText(String.valueOf(blockSize));
            blockScale.addListener(SWT.Selection, e -> blockLabel.setText(String.valueOf(blockScale.getSelection())));

            // C Value
            new Label(dialog, SWT.NONE).setText("C (constant):");
            Scale cScale = new Scale(dialog, SWT.HORIZONTAL);
            cScale.setMinimum(0);
            cScale.setMaximum(50);
            cScale.setSelection(cValue);
            cScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label cLabel = new Label(dialog, SWT.NONE);
            cLabel.setText(String.valueOf(cValue));
            cScale.addListener(SWT.Selection, e -> cLabel.setText(String.valueOf(cScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                maxValue = maxScale.getSelection();
                methodIndex = methodCombo.getSelectionIndex();
                typeIndex = typeCombo.getSelectionIndex();
                blockSize = blockScale.getSelection();
                cValue = cScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // CLAHE (Contrast Limited Adaptive Histogram Equalization) node
    static class CLAHENode extends ProcessingNode {
        private static final String[] COLOR_MODES = {"LAB", "HSV", "Grayscale"};

        private double clipLimit = 2.0;
        private int tileSize = 8;
        private int colorModeIndex = 0; // LAB

        public CLAHENode(Display display, Shell shell, int x, int y) {
            super(display, shell, "CLAHE", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) {
                return input;
            }

            CLAHE clahe = Imgproc.createCLAHE(clipLimit, new Size(tileSize, tileSize));
            Mat output = new Mat();

            if (colorModeIndex == 2) { // Grayscale
                Mat gray = new Mat();
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                Mat result = new Mat();
                clahe.apply(gray, result);
                Imgproc.cvtColor(result, output, Imgproc.COLOR_GRAY2BGR);
                gray.release();
                result.release();
            } else if (colorModeIndex == 0) { // LAB
                Mat lab = new Mat();
                Imgproc.cvtColor(input, lab, Imgproc.COLOR_BGR2Lab);
                java.util.List<Mat> channels = new java.util.ArrayList<>();
                Core.split(lab, channels);
                Mat lChannel = new Mat();
                clahe.apply(channels.get(0), lChannel);
                channels.set(0, lChannel);
                Core.merge(channels, lab);
                Imgproc.cvtColor(lab, output, Imgproc.COLOR_Lab2BGR);
                for (Mat ch : channels) ch.release();
                lab.release();
            } else { // HSV
                Mat hsv = new Mat();
                Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
                java.util.List<Mat> channels = new java.util.ArrayList<>();
                Core.split(hsv, channels);
                Mat vChannel = new Mat();
                clahe.apply(channels.get(2), vChannel);
                channels.set(2, vChannel);
                Core.merge(channels, hsv);
                Imgproc.cvtColor(hsv, output, Imgproc.COLOR_HSV2BGR);
                for (Mat ch : channels) ch.release();
                hsv.release();
            }

            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.createCLAHE(clipLimit, tileGridSize)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("CLAHE Properties");
            dialog.setLayout(new GridLayout(3, false));

            Label sigLabel = new Label(dialog, SWT.NONE);
            sigLabel.setText(getDescription());
            sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
            GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sigGd.horizontalSpan = 3;
            sigLabel.setLayoutData(sigGd);

            Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
            GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
            sepGd.horizontalSpan = 3;
            sep.setLayoutData(sepGd);

            // Clip Limit
            new Label(dialog, SWT.NONE).setText("Clip Limit:");
            Scale clipScale = new Scale(dialog, SWT.HORIZONTAL);
            clipScale.setMinimum(10);
            clipScale.setMaximum(400);
            clipScale.setSelection((int)(clipLimit * 10));
            clipScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label clipLabel = new Label(dialog, SWT.NONE);
            clipLabel.setText(String.format("%.1f", clipLimit));
            clipScale.addListener(SWT.Selection, e -> {
                double val = clipScale.getSelection() / 10.0;
                clipLabel.setText(String.format("%.1f", val));
            });

            // Tile Size
            new Label(dialog, SWT.NONE).setText("Tile Size:");
            Scale tileScale = new Scale(dialog, SWT.HORIZONTAL);
            tileScale.setMinimum(2);
            tileScale.setMaximum(32);
            tileScale.setSelection(tileSize);
            tileScale.setLayoutData(new GridData(200, SWT.DEFAULT));

            Label tileLabel = new Label(dialog, SWT.NONE);
            tileLabel.setText(String.valueOf(tileSize));
            tileScale.addListener(SWT.Selection, e -> tileLabel.setText(String.valueOf(tileScale.getSelection())));

            // Color Mode
            new Label(dialog, SWT.NONE).setText("Apply to:");
            Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            modeCombo.setItems(COLOR_MODES);
            modeCombo.select(colorModeIndex);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            modeCombo.setLayoutData(comboGd);

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                clipLimit = clipScale.getSelection() / 10.0;
                tileSize = tileScale.getSelection();
                colorModeIndex = modeCombo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Color In Range node - HSV/BGR color filtering
    static class ColorInRangeNode extends ProcessingNode {
        private boolean useHSV = true;
        private int hLow = 0, hHigh = 179;
        private int sLow = 0, sHigh = 255;
        private int vLow = 0, vHigh = 255;
        private int outputMode = 0; // 0=mask, 1=masked, 2=inverse

        private static final String[] OUTPUT_MODES = {"Mask Only", "Keep In-Range", "Keep Out-of-Range"};

        public ColorInRangeNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Color In Range", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) return input;

            // Ensure input is color
            Mat colorInput = input;
            if (input.channels() == 1) {
                colorInput = new Mat();
                Imgproc.cvtColor(input, colorInput, Imgproc.COLOR_GRAY2BGR);
            }

            // Convert to HSV if needed
            Mat converted = new Mat();
            if (useHSV) {
                Imgproc.cvtColor(colorInput, converted, Imgproc.COLOR_BGR2HSV);
            } else {
                converted = colorInput.clone();
            }

            // Create lower and upper bounds
            Scalar lower = new Scalar(hLow, sLow, vLow);
            Scalar upper = new Scalar(hHigh, sHigh, vHigh);

            // Create mask
            Mat mask = new Mat();
            Core.inRange(converted, lower, upper, mask);

            Mat result = new Mat();
            switch (outputMode) {
                case 0: // Mask only
                    Imgproc.cvtColor(mask, result, Imgproc.COLOR_GRAY2BGR);
                    break;
                case 1: // Keep in-range
                    result = new Mat();
                    colorInput.copyTo(result, mask);
                    break;
                case 2: // Keep out-of-range (inverse)
                    Mat invMask = new Mat();
                    Core.bitwise_not(mask, invMask);
                    result = new Mat();
                    colorInput.copyTo(result, invMask);
                    break;
                default:
                    Imgproc.cvtColor(mask, result, Imgproc.COLOR_GRAY2BGR);
            }

            return result;
        }

        @Override
        public String getDescription() {
            return "cv2.inRange(src, lowerb, upperb)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Color In Range Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Color space checkbox
            Button hsvCheck = new Button(dialog, SWT.CHECK);
            hsvCheck.setText("Use HSV (uncheck for BGR)");
            hsvCheck.setSelection(useHSV);
            GridData checkGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            checkGd.horizontalSpan = 3;
            hsvCheck.setLayoutData(checkGd);

            // H/B Low
            new Label(dialog, SWT.NONE).setText(useHSV ? "H Low:" : "B Low:");
            Scale hLowScale = new Scale(dialog, SWT.HORIZONTAL);
            hLowScale.setMinimum(0);
            hLowScale.setMaximum(255);
            hLowScale.setSelection(hLow);
            hLowScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label hLowLabel = new Label(dialog, SWT.NONE);
            hLowLabel.setText(String.valueOf(hLow));
            hLowScale.addListener(SWT.Selection, e -> hLowLabel.setText(String.valueOf(hLowScale.getSelection())));

            // H/B High
            new Label(dialog, SWT.NONE).setText(useHSV ? "H High:" : "B High:");
            Scale hHighScale = new Scale(dialog, SWT.HORIZONTAL);
            hHighScale.setMinimum(0);
            hHighScale.setMaximum(255);
            hHighScale.setSelection(hHigh);
            hHighScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label hHighLabel = new Label(dialog, SWT.NONE);
            hHighLabel.setText(String.valueOf(hHigh));
            hHighScale.addListener(SWT.Selection, e -> hHighLabel.setText(String.valueOf(hHighScale.getSelection())));

            // S/G Low
            new Label(dialog, SWT.NONE).setText(useHSV ? "S Low:" : "G Low:");
            Scale sLowScale = new Scale(dialog, SWT.HORIZONTAL);
            sLowScale.setMinimum(0);
            sLowScale.setMaximum(255);
            sLowScale.setSelection(sLow);
            sLowScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label sLowLabel = new Label(dialog, SWT.NONE);
            sLowLabel.setText(String.valueOf(sLow));
            sLowScale.addListener(SWT.Selection, e -> sLowLabel.setText(String.valueOf(sLowScale.getSelection())));

            // S/G High
            new Label(dialog, SWT.NONE).setText(useHSV ? "S High:" : "G High:");
            Scale sHighScale = new Scale(dialog, SWT.HORIZONTAL);
            sHighScale.setMinimum(0);
            sHighScale.setMaximum(255);
            sHighScale.setSelection(sHigh);
            sHighScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label sHighLabel = new Label(dialog, SWT.NONE);
            sHighLabel.setText(String.valueOf(sHigh));
            sHighScale.addListener(SWT.Selection, e -> sHighLabel.setText(String.valueOf(sHighScale.getSelection())));

            // V/R Low
            new Label(dialog, SWT.NONE).setText(useHSV ? "V Low:" : "R Low:");
            Scale vLowScale = new Scale(dialog, SWT.HORIZONTAL);
            vLowScale.setMinimum(0);
            vLowScale.setMaximum(255);
            vLowScale.setSelection(vLow);
            vLowScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label vLowLabel = new Label(dialog, SWT.NONE);
            vLowLabel.setText(String.valueOf(vLow));
            vLowScale.addListener(SWT.Selection, e -> vLowLabel.setText(String.valueOf(vLowScale.getSelection())));

            // V/R High
            new Label(dialog, SWT.NONE).setText(useHSV ? "V High:" : "R High:");
            Scale vHighScale = new Scale(dialog, SWT.HORIZONTAL);
            vHighScale.setMinimum(0);
            vHighScale.setMaximum(255);
            vHighScale.setSelection(vHigh);
            vHighScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label vHighLabel = new Label(dialog, SWT.NONE);
            vHighLabel.setText(String.valueOf(vHigh));
            vHighScale.addListener(SWT.Selection, e -> vHighLabel.setText(String.valueOf(vHighScale.getSelection())));

            // Output mode
            new Label(dialog, SWT.NONE).setText("Output:");
            Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            modeCombo.setItems(OUTPUT_MODES);
            modeCombo.select(outputMode);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            modeCombo.setLayoutData(comboGd);

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                useHSV = hsvCheck.getSelection();
                hLow = hLowScale.getSelection();
                hHigh = hHighScale.getSelection();
                sLow = sLowScale.getSelection();
                sHigh = sHighScale.getSelection();
                vLow = vLowScale.getSelection();
                vHigh = vHighScale.getSelection();
                outputMode = modeCombo.getSelectionIndex();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Mean Shift Filter node - color segmentation
    static class MeanShiftFilterNode extends ProcessingNode {
        private int spatialRadius = 20;
        private int colorRadius = 40;
        private int maxLevel = 1;

        public MeanShiftFilterNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Mean Shift", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) return input;

            // Ensure input is color
            Mat colorInput = input;
            if (input.channels() == 1) {
                colorInput = new Mat();
                Imgproc.cvtColor(input, colorInput, Imgproc.COLOR_GRAY2BGR);
            }

            Mat output = new Mat();
            Imgproc.pyrMeanShiftFiltering(colorInput, output, spatialRadius, colorRadius, maxLevel);
            return output;
        }

        @Override
        public String getDescription() {
            return "cv2.pyrMeanShiftFiltering(src, sp, sr)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Mean Shift Filter Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Spatial Radius
            new Label(dialog, SWT.NONE).setText("Spatial Radius:");
            Scale spatialScale = new Scale(dialog, SWT.HORIZONTAL);
            spatialScale.setMinimum(1);
            spatialScale.setMaximum(100);
            spatialScale.setSelection(spatialRadius);
            spatialScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label spatialLabel = new Label(dialog, SWT.NONE);
            spatialLabel.setText(String.valueOf(spatialRadius));
            spatialScale.addListener(SWT.Selection, e -> spatialLabel.setText(String.valueOf(spatialScale.getSelection())));

            // Color Radius
            new Label(dialog, SWT.NONE).setText("Color Radius:");
            Scale colorScale = new Scale(dialog, SWT.HORIZONTAL);
            colorScale.setMinimum(1);
            colorScale.setMaximum(100);
            colorScale.setSelection(colorRadius);
            colorScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label colorLabel = new Label(dialog, SWT.NONE);
            colorLabel.setText(String.valueOf(colorRadius));
            colorScale.addListener(SWT.Selection, e -> colorLabel.setText(String.valueOf(colorScale.getSelection())));

            // Max Level
            new Label(dialog, SWT.NONE).setText("Max Pyramid Level:");
            Scale levelScale = new Scale(dialog, SWT.HORIZONTAL);
            levelScale.setMinimum(0);
            levelScale.setMaximum(4);
            levelScale.setSelection(maxLevel);
            levelScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label levelLabel = new Label(dialog, SWT.NONE);
            levelLabel.setText(String.valueOf(maxLevel));
            levelScale.addListener(SWT.Selection, e -> levelLabel.setText(String.valueOf(levelScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                spatialRadius = spatialScale.getSelection();
                colorRadius = colorScale.getSelection();
                maxLevel = levelScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Hough Circles detection node
    static class HoughCirclesNode extends ProcessingNode {
        private int minDist = 50;
        private int param1 = 100;  // Canny high threshold
        private int param2 = 30;   // Accumulator threshold
        private int minRadius = 10;
        private int maxRadius = 100;
        private int thickness = 2;
        private boolean drawCenter = true;
        private int colorR = 0, colorG = 255, colorB = 0;

        public HoughCirclesNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Hough Circles", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) return input;

            // Convert to grayscale for detection
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Apply Gaussian blur to reduce noise
            Imgproc.GaussianBlur(gray, gray, new org.opencv.core.Size(9, 9), 2);

            // Create output image (color)
            Mat result = new Mat();
            if (input.channels() == 1) {
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }

            // Detect circles
            Mat circles = new Mat();
            Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1, minDist,
                param1, param2, minRadius, maxRadius);

            // Draw circles
            Scalar color = new Scalar(colorB, colorG, colorR);
            for (int i = 0; i < circles.cols(); i++) {
                double[] c = circles.get(0, i);
                org.opencv.core.Point center = new org.opencv.core.Point(c[0], c[1]);
                int radius = (int) Math.round(c[2]);

                // Draw circle outline
                Imgproc.circle(result, center, radius, color, thickness);

                // Draw center point
                if (drawCenter) {
                    Imgproc.circle(result, center, 2, color, 3);
                }
            }

            return result;
        }

        @Override
        public String getDescription() {
            return "cv2.HoughCircles(image, method, dp, minDist)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Hough Circles Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Min Distance
            new Label(dialog, SWT.NONE).setText("Min Distance:");
            Scale distScale = new Scale(dialog, SWT.HORIZONTAL);
            distScale.setMinimum(1);
            distScale.setMaximum(200);
            distScale.setSelection(minDist);
            distScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label distLabel = new Label(dialog, SWT.NONE);
            distLabel.setText(String.valueOf(minDist));
            distScale.addListener(SWT.Selection, e -> distLabel.setText(String.valueOf(distScale.getSelection())));

            // Param1 (Canny threshold)
            new Label(dialog, SWT.NONE).setText("Canny Threshold:");
            Scale p1Scale = new Scale(dialog, SWT.HORIZONTAL);
            p1Scale.setMinimum(1);
            p1Scale.setMaximum(300);
            p1Scale.setSelection(param1);
            p1Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label p1Label = new Label(dialog, SWT.NONE);
            p1Label.setText(String.valueOf(param1));
            p1Scale.addListener(SWT.Selection, e -> p1Label.setText(String.valueOf(p1Scale.getSelection())));

            // Param2 (Accumulator threshold)
            new Label(dialog, SWT.NONE).setText("Accum Threshold:");
            Scale p2Scale = new Scale(dialog, SWT.HORIZONTAL);
            p2Scale.setMinimum(1);
            p2Scale.setMaximum(100);
            p2Scale.setSelection(param2);
            p2Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label p2Label = new Label(dialog, SWT.NONE);
            p2Label.setText(String.valueOf(param2));
            p2Scale.addListener(SWT.Selection, e -> p2Label.setText(String.valueOf(p2Scale.getSelection())));

            // Min Radius
            new Label(dialog, SWT.NONE).setText("Min Radius:");
            Scale minRScale = new Scale(dialog, SWT.HORIZONTAL);
            minRScale.setMinimum(0);
            minRScale.setMaximum(200);
            minRScale.setSelection(minRadius);
            minRScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label minRLabel = new Label(dialog, SWT.NONE);
            minRLabel.setText(String.valueOf(minRadius));
            minRScale.addListener(SWT.Selection, e -> minRLabel.setText(String.valueOf(minRScale.getSelection())));

            // Max Radius
            new Label(dialog, SWT.NONE).setText("Max Radius:");
            Scale maxRScale = new Scale(dialog, SWT.HORIZONTAL);
            maxRScale.setMinimum(0);
            maxRScale.setMaximum(500);
            maxRScale.setSelection(maxRadius);
            maxRScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label maxRLabel = new Label(dialog, SWT.NONE);
            maxRLabel.setText(String.valueOf(maxRadius));
            maxRScale.addListener(SWT.Selection, e -> maxRLabel.setText(String.valueOf(maxRScale.getSelection())));

            // Thickness
            new Label(dialog, SWT.NONE).setText("Line Thickness:");
            Scale thickScale = new Scale(dialog, SWT.HORIZONTAL);
            thickScale.setMinimum(1);
            thickScale.setMaximum(10);
            thickScale.setSelection(thickness);
            thickScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label thickLabel = new Label(dialog, SWT.NONE);
            thickLabel.setText(String.valueOf(thickness));
            thickScale.addListener(SWT.Selection, e -> thickLabel.setText(String.valueOf(thickScale.getSelection())));

            // Draw center checkbox
            Button centerCheck = new Button(dialog, SWT.CHECK);
            centerCheck.setText("Draw Center Point");
            centerCheck.setSelection(drawCenter);
            GridData checkGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            checkGd.horizontalSpan = 3;
            centerCheck.setLayoutData(checkGd);

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                minDist = distScale.getSelection();
                param1 = p1Scale.getSelection();
                param2 = p2Scale.getSelection();
                minRadius = minRScale.getSelection();
                maxRadius = maxRScale.getSelection();
                thickness = thickScale.getSelection();
                drawCenter = centerCheck.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Hough Lines detection node
    static class HoughLinesNode extends ProcessingNode {
        private int threshold = 50;
        private int minLineLength = 50;
        private int maxLineGap = 10;
        private int cannyThresh1 = 50;
        private int cannyThresh2 = 150;
        private int thickness = 2;
        private int colorR = 255, colorG = 0, colorB = 0;

        public HoughLinesNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Hough Lines", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) return input;

            // Convert to grayscale for detection
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Apply Canny edge detection
            Mat edges = new Mat();
            Imgproc.Canny(gray, edges, cannyThresh1, cannyThresh2);

            // Create output image (color)
            Mat result = new Mat();
            if (input.channels() == 1) {
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }

            // Detect lines using probabilistic Hough transform
            Mat lines = new Mat();
            Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, threshold, minLineLength, maxLineGap);

            // Draw lines
            Scalar color = new Scalar(colorB, colorG, colorR);
            for (int i = 0; i < lines.rows(); i++) {
                double[] l = lines.get(i, 0);
                Imgproc.line(result,
                    new org.opencv.core.Point(l[0], l[1]),
                    new org.opencv.core.Point(l[2], l[3]),
                    color, thickness);
            }

            return result;
        }

        @Override
        public String getDescription() {
            return "cv2.HoughLinesP(image, rho, theta, threshold)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Hough Lines Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Canny Threshold 1
            new Label(dialog, SWT.NONE).setText("Canny Thresh 1:");
            Scale c1Scale = new Scale(dialog, SWT.HORIZONTAL);
            c1Scale.setMinimum(0);
            c1Scale.setMaximum(255);
            c1Scale.setSelection(cannyThresh1);
            c1Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label c1Label = new Label(dialog, SWT.NONE);
            c1Label.setText(String.valueOf(cannyThresh1));
            c1Scale.addListener(SWT.Selection, e -> c1Label.setText(String.valueOf(c1Scale.getSelection())));

            // Canny Threshold 2
            new Label(dialog, SWT.NONE).setText("Canny Thresh 2:");
            Scale c2Scale = new Scale(dialog, SWT.HORIZONTAL);
            c2Scale.setMinimum(0);
            c2Scale.setMaximum(255);
            c2Scale.setSelection(cannyThresh2);
            c2Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label c2Label = new Label(dialog, SWT.NONE);
            c2Label.setText(String.valueOf(cannyThresh2));
            c2Scale.addListener(SWT.Selection, e -> c2Label.setText(String.valueOf(c2Scale.getSelection())));

            // Threshold
            new Label(dialog, SWT.NONE).setText("Hough Threshold:");
            Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
            threshScale.setMinimum(1);
            threshScale.setMaximum(200);
            threshScale.setSelection(threshold);
            threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label threshLabel = new Label(dialog, SWT.NONE);
            threshLabel.setText(String.valueOf(threshold));
            threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

            // Min Line Length
            new Label(dialog, SWT.NONE).setText("Min Line Length:");
            Scale minLenScale = new Scale(dialog, SWT.HORIZONTAL);
            minLenScale.setMinimum(1);
            minLenScale.setMaximum(200);
            minLenScale.setSelection(minLineLength);
            minLenScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label minLenLabel = new Label(dialog, SWT.NONE);
            minLenLabel.setText(String.valueOf(minLineLength));
            minLenScale.addListener(SWT.Selection, e -> minLenLabel.setText(String.valueOf(minLenScale.getSelection())));

            // Max Line Gap
            new Label(dialog, SWT.NONE).setText("Max Line Gap:");
            Scale gapScale = new Scale(dialog, SWT.HORIZONTAL);
            gapScale.setMinimum(1);
            gapScale.setMaximum(100);
            gapScale.setSelection(maxLineGap);
            gapScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label gapLabel = new Label(dialog, SWT.NONE);
            gapLabel.setText(String.valueOf(maxLineGap));
            gapScale.addListener(SWT.Selection, e -> gapLabel.setText(String.valueOf(gapScale.getSelection())));

            // Thickness
            new Label(dialog, SWT.NONE).setText("Line Thickness:");
            Scale thickScale = new Scale(dialog, SWT.HORIZONTAL);
            thickScale.setMinimum(1);
            thickScale.setMaximum(10);
            thickScale.setSelection(thickness);
            thickScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label thickLabel = new Label(dialog, SWT.NONE);
            thickLabel.setText(String.valueOf(thickness));
            thickScale.addListener(SWT.Selection, e -> thickLabel.setText(String.valueOf(thickScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                cannyThresh1 = c1Scale.getSelection();
                cannyThresh2 = c2Scale.getSelection();
                threshold = threshScale.getSelection();
                minLineLength = minLenScale.getSelection();
                maxLineGap = gapScale.getSelection();
                thickness = thickScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Contours detection node
    static class ContoursNode extends ProcessingNode {
        private int thresholdValue = 127;
        private int retrievalMode = 0; // 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE
        private int thickness = 2;
        private int colorR = 0, colorG = 255, colorB = 0;

        private static final String[] RETRIEVAL_MODES = {"External", "List", "Two-level", "Tree"};
        private static final int[] RETRIEVAL_VALUES = {
            Imgproc.RETR_EXTERNAL, Imgproc.RETR_LIST, Imgproc.RETR_CCOMP, Imgproc.RETR_TREE
        };

        public ContoursNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Contours", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) return input;

            // Convert to grayscale
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Apply threshold
            Mat binary = new Mat();
            Imgproc.threshold(gray, binary, thresholdValue, 255, Imgproc.THRESH_BINARY);

            // Find contours
            java.util.List<org.opencv.core.MatOfPoint> contours = new java.util.ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy,
                RETRIEVAL_VALUES[retrievalMode], Imgproc.CHAIN_APPROX_SIMPLE);

            // Create output image (color)
            Mat result = new Mat();
            if (input.channels() == 1) {
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }

            // Draw contours
            Scalar color = new Scalar(colorB, colorG, colorR);
            Imgproc.drawContours(result, contours, -1, color, thickness);

            return result;
        }

        @Override
        public String getDescription() {
            return "cv2.findContours(image, mode, method)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Contours Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Threshold
            new Label(dialog, SWT.NONE).setText("Threshold:");
            Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
            threshScale.setMinimum(0);
            threshScale.setMaximum(255);
            threshScale.setSelection(thresholdValue);
            threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label threshLabel = new Label(dialog, SWT.NONE);
            threshLabel.setText(String.valueOf(thresholdValue));
            threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

            // Retrieval Mode
            new Label(dialog, SWT.NONE).setText("Retrieval Mode:");
            Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            modeCombo.setItems(RETRIEVAL_MODES);
            modeCombo.select(retrievalMode);
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            modeCombo.setLayoutData(comboGd);

            // Thickness
            new Label(dialog, SWT.NONE).setText("Line Thickness:");
            Scale thickScale = new Scale(dialog, SWT.HORIZONTAL);
            thickScale.setMinimum(1);
            thickScale.setMaximum(10);
            thickScale.setSelection(thickness);
            thickScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label thickLabel = new Label(dialog, SWT.NONE);
            thickLabel.setText(String.valueOf(thickness));
            thickScale.addListener(SWT.Selection, e -> thickLabel.setText(String.valueOf(thickScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                thresholdValue = threshScale.getSelection();
                retrievalMode = modeCombo.getSelectionIndex();
                thickness = thickScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
            Point cursor = shell.getDisplay().getCursorLocation();
            dialog.setLocation(cursor.x, cursor.y);
            dialog.open();
        }
    }

    // Harris Corners detection node
    static class HarrisCornersNode extends ProcessingNode {
        private int blockSize = 2;
        private int ksize = 3;
        private int thresholdPercent = 1; // 0.01 * 100
        private int markerSize = 5;
        private int colorR = 255, colorG = 0, colorB = 0;

        public HarrisCornersNode(Display display, Shell shell, int x, int y) {
            super(display, shell, "Harris Corners", x, y);
        }

        @Override
        public Mat process(Mat input) {
            if (!enabled || input == null || input.empty()) return input;

            // Convert to grayscale
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Convert to float
            Mat grayFloat = new Mat();
            gray.convertTo(grayFloat, CvType.CV_32F);

            // Apply Harris corner detection
            Mat harris = new Mat();
            Imgproc.cornerHarris(grayFloat, harris, blockSize, ksize, 0.04);

            // Normalize and convert to byte for thresholding
            Mat harrisNorm = new Mat();
            Core.normalize(harris, harrisNorm, 0, 255, Core.NORM_MINMAX);
            Mat harrisNormScaled = new Mat();
            harrisNorm.convertTo(harrisNormScaled, CvType.CV_8U);

            // Create output image (color)
            Mat result = new Mat();
            if (input.channels() == 1) {
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }

            // Find and draw corners
            Scalar color = new Scalar(colorB, colorG, colorR);
            double threshold = thresholdPercent * 2.55; // Convert percent to 0-255 range

            for (int i = 0; i < harrisNormScaled.rows(); i++) {
                for (int j = 0; j < harrisNormScaled.cols(); j++) {
                    if (harrisNormScaled.get(i, j)[0] > threshold) {
                        Imgproc.circle(result, new org.opencv.core.Point(j, i), markerSize, color, -1);
                    }
                }
            }

            return result;
        }

        @Override
        public String getDescription() {
            return "cv2.cornerHarris(src, blockSize, ksize, k)";
        }

        @Override
        public void showPropertiesDialog() {
            Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
            dialog.setText("Harris Corners Properties");
            dialog.setLayout(new GridLayout(3, false));

            // Block Size
            new Label(dialog, SWT.NONE).setText("Block Size:");
            Scale blockScale = new Scale(dialog, SWT.HORIZONTAL);
            blockScale.setMinimum(2);
            blockScale.setMaximum(10);
            blockScale.setSelection(blockSize);
            blockScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label blockLabel = new Label(dialog, SWT.NONE);
            blockLabel.setText(String.valueOf(blockSize));
            blockScale.addListener(SWT.Selection, e -> blockLabel.setText(String.valueOf(blockScale.getSelection())));

            // Aperture Size (ksize)
            new Label(dialog, SWT.NONE).setText("Aperture Size:");
            Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
            ksizeCombo.setItems(new String[]{"3", "5", "7"});
            ksizeCombo.select(ksize == 3 ? 0 : (ksize == 5 ? 1 : 2));
            GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
            comboGd.horizontalSpan = 2;
            ksizeCombo.setLayoutData(comboGd);

            // Threshold
            new Label(dialog, SWT.NONE).setText("Threshold %:");
            Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
            threshScale.setMinimum(1);
            threshScale.setMaximum(100);
            threshScale.setSelection(thresholdPercent);
            threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label threshLabel = new Label(dialog, SWT.NONE);
            threshLabel.setText(String.valueOf(thresholdPercent));
            threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

            // Marker Size
            new Label(dialog, SWT.NONE).setText("Marker Size:");
            Scale markerScale = new Scale(dialog, SWT.HORIZONTAL);
            markerScale.setMinimum(1);
            markerScale.setMaximum(15);
            markerScale.setSelection(markerSize);
            markerScale.setLayoutData(new GridData(200, SWT.DEFAULT));
            Label markerLabel = new Label(dialog, SWT.NONE);
            markerLabel.setText(String.valueOf(markerSize));
            markerScale.addListener(SWT.Selection, e -> markerLabel.setText(String.valueOf(markerScale.getSelection())));

            Composite buttonComp = new Composite(dialog, SWT.NONE);
            buttonComp.setLayout(new GridLayout(2, true));
            GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
            gd.horizontalSpan = 3;
            buttonComp.setLayoutData(gd);

            Button okBtn = new Button(buttonComp, SWT.PUSH);
            okBtn.setText("OK");
            okBtn.addListener(SWT.Selection, e -> {
                blockSize = blockScale.getSelection();
                int idx = ksizeCombo.getSelectionIndex();
                ksize = idx == 0 ? 3 : (idx == 1 ? 5 : 7);
                thresholdPercent = threshScale.getSelection();
                markerSize = markerScale.getSelection();
                dialog.dispose();
                notifyChanged();
            });

            Button cancelBtn = new Button(buttonComp, SWT.PUSH);
            cancelBtn.setText("Cancel");
            cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

            dialog.pack();
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
