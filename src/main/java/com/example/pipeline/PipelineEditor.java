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

// Import extracted node classes
import com.example.pipeline.nodes.*;
import com.example.pipeline.model.*;
import com.example.pipeline.registry.NodeRegistry;

public class PipelineEditor {

    // Static initialization - register all node types with NodeRegistry
    static {
        registerNodes();
    }

    private static void registerNodes() {
        // Basic processing nodes
        NodeRegistry.register("Grayscale", "Basic", GrayscaleNode.class);
        NodeRegistry.register("Invert", "Basic", InvertNode.class);
        NodeRegistry.register("Gain", "Basic", GainNode.class);
        NodeRegistry.register("Threshold", "Basic", ThresholdNode.class);
        NodeRegistry.register("AdaptiveThreshold", "Basic", AdaptiveThresholdNode.class);
        NodeRegistry.register("CLAHE", "Basic", CLAHENode.class);
        NodeRegistry.register("ColorInRange", "Basic", ColorInRangeNode.class);
        NodeRegistry.register("BitPlanesGrayscale", "Basic", BitPlanesGrayscaleNode.class);
        NodeRegistry.register("BitPlanesColor", "Basic", BitPlanesColorNode.class);

        // Blur nodes
        NodeRegistry.register("GaussianBlur", "Blur", GaussianBlurNode.class);
        NodeRegistry.register("MedianBlur", "Blur", MedianBlurNode.class);
        NodeRegistry.register("BilateralFilter", "Blur", BilateralFilterNode.class);
        NodeRegistry.register("MeanShift", "Blur", MeanShiftFilterNode.class);

        // Edge detection nodes
        NodeRegistry.register("CannyEdge", "Edge Detection", CannyEdgeNode.class);
        NodeRegistry.register("Laplacian", "Edge Detection", LaplacianNode.class);
        NodeRegistry.register("Sobel", "Edge Detection", SobelNode.class);
        NodeRegistry.register("Scharr", "Edge Detection", ScharrNode.class);

        // Morphological nodes
        NodeRegistry.register("Erode", "Morphological", ErodeNode.class);
        NodeRegistry.register("Dilate", "Morphological", DilateNode.class);
        NodeRegistry.register("MorphOpen", "Morphological", MorphOpenNode.class);
        NodeRegistry.register("MorphClose", "Morphological", MorphCloseNode.class);

        // Detection nodes
        NodeRegistry.register("HoughCircles", "Detection", HoughCirclesNode.class);
        NodeRegistry.register("HoughLines", "Detection", HoughLinesNode.class);
        NodeRegistry.register("Contours", "Detection", ContoursNode.class);
        NodeRegistry.register("HarrisCorners", "Detection", HarrisCornersNode.class);
        NodeRegistry.register("ShiTomasi", "Detection", ShiTomasiCornersNode.class);
        NodeRegistry.register("BlobDetector", "Detection", BlobDetectorNode.class);
        NodeRegistry.register("ORBFeatures", "Detection", ORBFeaturesNode.class);
        NodeRegistry.register("SIFTFeatures", "Detection", SIFTFeaturesNode.class);
        NodeRegistry.register("ConnectedComponents", "Detection", ConnectedComponentsNode.class);

        // Transform nodes
        NodeRegistry.register("WarpAffine", "Transform", WarpAffineNode.class);

        // Filter nodes
        NodeRegistry.register("FFTFilter", "Filter", FFTFilterNode.class);
    }

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

        // Show splash screen immediately
        Shell splash = showSplashScreen();

        shell = new Shell(display);
        shell.setText("OpenCV Pipeline Editor");
        shell.setSize(1400, 800);
        shell.setLayout(new GridLayout(2, false));

        // Center main window on screen
        Rectangle screenBounds = display.getPrimaryMonitor().getBounds();
        Rectangle shellBounds = shell.getBounds();
        int x = screenBounds.x + (screenBounds.width - shellBounds.width) / 2;
        int y = screenBounds.y + (screenBounds.height - shellBounds.height) / 2;
        shell.setLocation(x, y);

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

        // Close splash and open main window
        if (splash != null && !splash.isDisposed()) {
            splash.close();
        }
        shell.open();

        while (!shell.isDisposed()) {
            if (!display.readAndDispatch()) {
                display.sleep();
            }
        }
        display.dispose();
    }

    private Shell showSplashScreen() {
        Shell splash = new Shell(display, SWT.NO_TRIM | SWT.ON_TOP);

        // Create splash content
        splash.setLayout(new GridLayout(1, false));
        splash.setBackground(new Color(45, 45, 48)); // Dark background

        // Title
        Label titleLabel = new Label(splash, SWT.CENTER);
        titleLabel.setText("OpenCV Pipeline Editor");
        titleLabel.setFont(new Font(display, "Arial", 24, SWT.BOLD));
        titleLabel.setForeground(new Color(255, 255, 255));
        titleLabel.setBackground(splash.getBackground());
        titleLabel.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, true, false));

        // Subtitle
        Label subtitleLabel = new Label(splash, SWT.CENTER);
        subtitleLabel.setText("Visual Image Processing");
        subtitleLabel.setFont(new Font(display, "Arial", 12, SWT.NORMAL));
        subtitleLabel.setForeground(new Color(180, 180, 180));
        subtitleLabel.setBackground(splash.getBackground());
        subtitleLabel.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, true, false));

        // Spacer
        Label spacer = new Label(splash, SWT.NONE);
        spacer.setBackground(splash.getBackground());
        GridData spacerGd = new GridData(SWT.FILL, SWT.FILL, true, true);
        spacerGd.heightHint = 20;
        spacer.setLayoutData(spacerGd);

        // Loading message
        Label loadingLabel = new Label(splash, SWT.CENTER);
        loadingLabel.setText("Loading...");
        loadingLabel.setFont(new Font(display, "Arial", 10, SWT.ITALIC));
        loadingLabel.setForeground(new Color(120, 120, 120));
        loadingLabel.setBackground(splash.getBackground());
        loadingLabel.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, true, false));

        // Size and center splash
        splash.setSize(350, 180);
        Rectangle screenBounds = display.getPrimaryMonitor().getBounds();
        Rectangle splashBounds = splash.getBounds();
        int x = screenBounds.x + (screenBounds.width - splashBounds.width) / 2;
        int y = screenBounds.y + (screenBounds.height - splashBounds.height) / 2;
        splash.setLocation(x, y);

        splash.open();

        // Process events to show splash immediately
        while (display.readAndDispatch()) {
            // Process all pending events
        }

        return splash;
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
                if (node instanceof FileSourceNode) {
                    ((FileSourceNode) node).getOverlayComposite().dispose();
                } else if (node instanceof WebcamSourceNode) {
                    ((WebcamSourceNode) node).getOverlayComposite().dispose();
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

                if ("FileSource".equals(type)) {
                    FileSourceNode node = new FileSourceNode(shell, display, canvas, x, y);
                    if (nodeObj.has("imagePath")) {
                        String imgPath = nodeObj.get("imagePath").getAsString();
                        node.setImagePath(imgPath);
                        node.loadMedia(imgPath);
                    }
                    if (nodeObj.has("fpsMode")) {
                        node.setFpsMode(nodeObj.get("fpsMode").getAsInt());
                    }
                    nodes.add(node);
                } else if ("WebcamSource".equals(type)) {
                    WebcamSourceNode node = new WebcamSourceNode(shell, display, canvas, x, y);
                    if (nodeObj.has("cameraIndex")) {
                        node.setCameraIndex(nodeObj.get("cameraIndex").getAsInt());
                    }
                    if (nodeObj.has("resolutionIndex")) {
                        node.setResolutionIndex(nodeObj.get("resolutionIndex").getAsInt());
                    }
                    if (nodeObj.has("mirrorHorizontal")) {
                        node.setMirrorHorizontal(nodeObj.get("mirrorHorizontal").getAsBoolean());
                    }
                    if (nodeObj.has("fpsIndex")) {
                        node.setFpsIndex(nodeObj.get("fpsIndex").getAsInt());
                    }
                    // Re-open camera with deserialized settings on background thread
                    new Thread(() -> node.openCamera()).start();
                    nodes.add(node);
                } else if ("Processing".equals(type)) {
                    String name = nodeObj.get("name").getAsString();
                    ProcessingNode node = createEffectNode(name, x, y);
                    if (node != null) {
                        // Load node-specific properties
                        if (node instanceof GaussianBlurNode) {
                            GaussianBlurNode gbn = (GaussianBlurNode) node;
                            if (nodeObj.has("kernelSizeX")) gbn.setKernelSizeX(nodeObj.get("kernelSizeX").getAsInt());
                            if (nodeObj.has("kernelSizeY")) gbn.setKernelSizeY(nodeObj.get("kernelSizeY").getAsInt());
                            if (nodeObj.has("sigmaX")) gbn.setSigmaX(nodeObj.get("sigmaX").getAsDouble());
                        } else if (node instanceof GrayscaleNode) {
                            GrayscaleNode gn = (GrayscaleNode) node;
                            if (nodeObj.has("conversionIndex")) {
                                gn.setConversionIndex(nodeObj.get("conversionIndex").getAsInt());
                            }
                        } else if (node instanceof ThresholdNode) {
                            ThresholdNode tn = (ThresholdNode) node;
                            if (nodeObj.has("threshValue")) tn.setThreshValue(nodeObj.get("threshValue").getAsInt());
                            if (nodeObj.has("maxValue")) tn.setMaxValue(nodeObj.get("maxValue").getAsInt());
                            if (nodeObj.has("typeIndex")) tn.setTypeIndex(nodeObj.get("typeIndex").getAsInt());
                            if (nodeObj.has("modifierIndex")) tn.setModifierIndex(nodeObj.get("modifierIndex").getAsInt());
                        } else if (node instanceof GainNode) {
                            GainNode gn = (GainNode) node;
                            if (nodeObj.has("gain")) gn.setGain(nodeObj.get("gain").getAsDouble());
                        } else if (node instanceof BitPlanesGrayscaleNode) {
                            BitPlanesGrayscaleNode bpn = (BitPlanesGrayscaleNode) node;
                            if (nodeObj.has("bitEnabled")) {
                                JsonArray arr = nodeObj.getAsJsonArray("bitEnabled");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitEnabled(j, arr.get(j).getAsBoolean());
                                }
                            }
                            if (nodeObj.has("bitGain")) {
                                JsonArray arr = nodeObj.getAsJsonArray("bitGain");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitGain(j, arr.get(j).getAsDouble());
                                }
                            }
                        } else if (node instanceof BitPlanesColorNode) {
                            BitPlanesColorNode bpn = (BitPlanesColorNode) node;
                            // Load Red channel
                            if (nodeObj.has("redBitEnabled")) {
                                JsonArray arr = nodeObj.getAsJsonArray("redBitEnabled");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitEnabled(0, j, arr.get(j).getAsBoolean());
                                }
                            }
                            if (nodeObj.has("redBitGain")) {
                                JsonArray arr = nodeObj.getAsJsonArray("redBitGain");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitGain(0, j, arr.get(j).getAsDouble());
                                }
                            }
                            // Load Green channel
                            if (nodeObj.has("greenBitEnabled")) {
                                JsonArray arr = nodeObj.getAsJsonArray("greenBitEnabled");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitEnabled(1, j, arr.get(j).getAsBoolean());
                                }
                            }
                            if (nodeObj.has("greenBitGain")) {
                                JsonArray arr = nodeObj.getAsJsonArray("greenBitGain");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitGain(1, j, arr.get(j).getAsDouble());
                                }
                            }
                            // Load Blue channel
                            if (nodeObj.has("blueBitEnabled")) {
                                JsonArray arr = nodeObj.getAsJsonArray("blueBitEnabled");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitEnabled(2, j, arr.get(j).getAsBoolean());
                                }
                            }
                            if (nodeObj.has("blueBitGain")) {
                                JsonArray arr = nodeObj.getAsJsonArray("blueBitGain");
                                for (int j = 0; j < 8 && j < arr.size(); j++) {
                                    bpn.setBitGain(2, j, arr.get(j).getAsDouble());
                                }
                            }
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
            if (node instanceof FileSourceNode) {
                ((FileSourceNode) node).getOverlayComposite().dispose();
            } else if (node instanceof WebcamSourceNode) {
                ((WebcamSourceNode) node).getOverlayComposite().dispose();
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

        createNodeButton(toolbar, "File Source", () -> addFileSourceNode());
        createNodeButton(toolbar, "Webcam Source", () -> addWebcamSourceNode());

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Basic Effects
        Label basicLabel = new Label(toolbar, SWT.NONE);
        basicLabel.setText("Basic:");
        basicLabel.setFont(boldFont);

        createNodeButton(toolbar, "Grayscale/Color", () -> addEffectNode("Grayscale"));
        createNodeButton(toolbar, "Invert", () -> addEffectNode("Invert"));
        createNodeButton(toolbar, "Gain", () -> addEffectNode("Gain"));
        createNodeButton(toolbar, "Threshold", () -> addEffectNode("Threshold"));
        createNodeButton(toolbar, "Adaptive Threshold", () -> addEffectNode("AdaptiveThreshold"));
        createNodeButton(toolbar, "CLAHE", () -> addEffectNode("CLAHE"));
        createNodeButton(toolbar, "Color In Range", () -> addEffectNode("ColorInRange"));
        createNodeButton(toolbar, "Bit Planes Grayscale", () -> addEffectNode("BitPlanesGrayscale"));
        createNodeButton(toolbar, "Bit Planes Color", () -> addEffectNode("BitPlanesColor"));

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
        createNodeButton(toolbar, "Shi-Tomasi", () -> addEffectNode("ShiTomasi"));
        createNodeButton(toolbar, "Blob Detector", () -> addEffectNode("BlobDetector"));
        createNodeButton(toolbar, "ORB Features", () -> addEffectNode("ORBFeatures"));
        createNodeButton(toolbar, "SIFT Features", () -> addEffectNode("SIFTFeatures"));
        createNodeButton(toolbar, "Connected Components", () -> addEffectNode("ConnectedComponents"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Transform
        Label transformLabel = new Label(toolbar, SWT.NONE);
        transformLabel.setText("Transform:");
        transformLabel.setFont(boldFont);

        createNodeButton(toolbar, "Warp Affine", () -> addEffectNode("WarpAffine"));

        // Separator
        new Label(toolbar, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Filter
        Label filterLabel = new Label(toolbar, SWT.NONE);
        filterLabel.setText("Filter:");
        filterLabel.setFont(boldFont);

        createNodeButton(toolbar, "FFT High-Pass", () -> addEffectNode("FFTFilter"));

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

                if (node instanceof FileSourceNode) {
                    nodeObj.addProperty("type", "FileSource");
                    FileSourceNode isn = (FileSourceNode) node;
                    if (isn.getImagePath() != null) {
                        nodeObj.addProperty("imagePath", isn.getImagePath());
                    }
                    nodeObj.addProperty("fpsMode", isn.getFpsMode());
                } else if (node instanceof WebcamSourceNode) {
                    nodeObj.addProperty("type", "WebcamSource");
                    WebcamSourceNode wsn = (WebcamSourceNode) node;
                    nodeObj.addProperty("cameraIndex", wsn.getCameraIndex());
                    nodeObj.addProperty("resolutionIndex", wsn.getResolutionIndex());
                    nodeObj.addProperty("mirrorHorizontal", wsn.isMirrorHorizontal());
                    nodeObj.addProperty("fpsIndex", wsn.getFpsIndex());
                } else if (node instanceof ProcessingNode) {
                    nodeObj.addProperty("type", "Processing");
                    nodeObj.addProperty("name", ((ProcessingNode) node).getName());

                    // Save node-specific properties
                    if (node instanceof GaussianBlurNode) {
                        GaussianBlurNode gbn = (GaussianBlurNode) node;
                        nodeObj.addProperty("kernelSizeX", gbn.getKernelSizeX());
                        nodeObj.addProperty("kernelSizeY", gbn.getKernelSizeY());
                        nodeObj.addProperty("sigmaX", gbn.getSigmaX());
                    } else if (node instanceof GrayscaleNode) {
                        GrayscaleNode gn = (GrayscaleNode) node;
                        nodeObj.addProperty("conversionIndex", gn.getConversionIndex());
                    } else if (node instanceof ThresholdNode) {
                        ThresholdNode tn = (ThresholdNode) node;
                        nodeObj.addProperty("threshValue", tn.getThreshValue());
                        nodeObj.addProperty("maxValue", tn.getMaxValue());
                        nodeObj.addProperty("typeIndex", tn.getTypeIndex());
                        nodeObj.addProperty("modifierIndex", tn.getModifierIndex());
                    } else if (node instanceof GainNode) {
                        GainNode gn = (GainNode) node;
                        nodeObj.addProperty("gain", gn.getGain());
                    } else if (node instanceof BitPlanesGrayscaleNode) {
                        BitPlanesGrayscaleNode bpn = (BitPlanesGrayscaleNode) node;
                        JsonArray enabledArray = new JsonArray();
                        JsonArray gainArray = new JsonArray();
                        for (int j = 0; j < 8; j++) {
                            enabledArray.add(bpn.getBitEnabled(j));
                            gainArray.add(bpn.getBitGain(j));
                        }
                        nodeObj.add("bitEnabled", enabledArray);
                        nodeObj.add("bitGain", gainArray);
                    } else if (node instanceof BitPlanesColorNode) {
                        BitPlanesColorNode bpn = (BitPlanesColorNode) node;
                        // Save Red channel
                        JsonArray redEnabled = new JsonArray();
                        JsonArray redGain = new JsonArray();
                        for (int j = 0; j < 8; j++) {
                            redEnabled.add(bpn.getBitEnabled(0, j));
                            redGain.add(bpn.getBitGain(0, j));
                        }
                        nodeObj.add("redBitEnabled", redEnabled);
                        nodeObj.add("redBitGain", redGain);
                        // Save Green channel
                        JsonArray greenEnabled = new JsonArray();
                        JsonArray greenGain = new JsonArray();
                        for (int j = 0; j < 8; j++) {
                            greenEnabled.add(bpn.getBitEnabled(1, j));
                            greenGain.add(bpn.getBitGain(1, j));
                        }
                        nodeObj.add("greenBitEnabled", greenEnabled);
                        nodeObj.add("greenBitGain", greenGain);
                        // Save Blue channel
                        JsonArray blueEnabled = new JsonArray();
                        JsonArray blueGain = new JsonArray();
                        for (int j = 0; j < 8; j++) {
                            blueEnabled.add(bpn.getBitEnabled(2, j));
                            blueGain.add(bpn.getBitGain(2, j));
                        }
                        nodeObj.add("blueBitEnabled", blueEnabled);
                        nodeObj.add("blueBitGain", blueGain);
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
                if (node instanceof FileSourceNode) {
                    ((FileSourceNode) node).saveThumbnailToCache(cacheDir);
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
                    if (node instanceof FileSourceNode) {
                        ((FileSourceNode) node).getOverlayComposite().dispose();
                    } else if (node instanceof WebcamSourceNode) {
                        ((WebcamSourceNode) node).getOverlayComposite().dispose();
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

                    if ("FileSource".equals(type)) {
                        FileSourceNode node = new FileSourceNode(shell, display, canvas, x, y);
                        if (nodeObj.has("imagePath")) {
                            String imgPath = nodeObj.get("imagePath").getAsString();
                            node.setImagePath(imgPath);
                            // Load the media (creates thumbnail and loads image for execution)
                            node.loadMedia(imgPath);
                        } else {
                        }
                        nodes.add(node);
                    } else if ("Processing".equals(type)) {
                        String name = nodeObj.get("name").getAsString();
                        ProcessingNode node = createEffectNode(name, x, y);
                        System.out.println("Created node for '" + name + "': " + (node != null ? node.getClass().getSimpleName() : "null"));
                        if (node != null) {
                            // Load node-specific properties
                            if (node instanceof GaussianBlurNode) {
                                GaussianBlurNode gbn = (GaussianBlurNode) node;
                                if (nodeObj.has("kernelSizeX")) gbn.setKernelSizeX(nodeObj.get("kernelSizeX").getAsInt());
                                if (nodeObj.has("kernelSizeY")) gbn.setKernelSizeY(nodeObj.get("kernelSizeY").getAsInt());
                                if (nodeObj.has("sigmaX")) gbn.setSigmaX(nodeObj.get("sigmaX").getAsDouble());
                            } else if (node instanceof GrayscaleNode) {
                                GrayscaleNode gn = (GrayscaleNode) node;
                                if (nodeObj.has("conversionIndex")) {
                                    int loadedIndex = nodeObj.get("conversionIndex").getAsInt();
                                    gn.setConversionIndex(loadedIndex);
                                } else {
                                }
                            } else if (node instanceof ThresholdNode) {
                                ThresholdNode tn = (ThresholdNode) node;
                                if (nodeObj.has("threshValue")) tn.setThreshValue(nodeObj.get("threshValue").getAsInt());
                                if (nodeObj.has("maxValue")) tn.setMaxValue(nodeObj.get("maxValue").getAsInt());
                                if (nodeObj.has("typeIndex")) tn.setTypeIndex(nodeObj.get("typeIndex").getAsInt());
                                if (nodeObj.has("modifierIndex")) tn.setModifierIndex(nodeObj.get("modifierIndex").getAsInt());
                            } else if (node instanceof GainNode) {
                                GainNode gn = (GainNode) node;
                                if (nodeObj.has("gain")) gn.setGain(nodeObj.get("gain").getAsDouble());
                            } else if (node instanceof BitPlanesGrayscaleNode) {
                                BitPlanesGrayscaleNode bpn = (BitPlanesGrayscaleNode) node;
                                System.out.println("Loading BitPlanesGrayscaleNode, has bitEnabled: " + nodeObj.has("bitEnabled") + ", has bitGain: " + nodeObj.has("bitGain"));
                                if (nodeObj.has("bitEnabled")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("bitEnabled");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitEnabled(j, arr.get(j).getAsBoolean());
                                    }
                                }
                                if (nodeObj.has("bitGain")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("bitGain");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitGain(j, arr.get(j).getAsDouble());
                                    }
                                }
                            } else if (node instanceof BitPlanesColorNode) {
                                System.out.println("Loading BitPlanesColorNode, has redBitEnabled: " + nodeObj.has("redBitEnabled"));
                                BitPlanesColorNode bpn = (BitPlanesColorNode) node;
                                // Load Red channel
                                if (nodeObj.has("redBitEnabled")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("redBitEnabled");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitEnabled(0, j, arr.get(j).getAsBoolean());
                                    }
                                }
                                if (nodeObj.has("redBitGain")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("redBitGain");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitGain(0, j, arr.get(j).getAsDouble());
                                    }
                                }
                                // Load Green channel
                                if (nodeObj.has("greenBitEnabled")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("greenBitEnabled");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitEnabled(1, j, arr.get(j).getAsBoolean());
                                    }
                                }
                                if (nodeObj.has("greenBitGain")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("greenBitGain");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitGain(1, j, arr.get(j).getAsDouble());
                                    }
                                }
                                // Load Blue channel
                                if (nodeObj.has("blueBitEnabled")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("blueBitEnabled");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitEnabled(2, j, arr.get(j).getAsBoolean());
                                    }
                                }
                                if (nodeObj.has("blueBitGain")) {
                                    JsonArray arr = nodeObj.getAsJsonArray("blueBitGain");
                                    for (int j = 0; j < 8 && j < arr.size(); j++) {
                                        bpn.setBitGain(2, j, arr.get(j).getAsDouble());
                                    }
                                }
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

        // Find source node (FileSourceNode or WebcamSourceNode with no incoming connections)
        PipelineNode sourceNode = null;
        for (PipelineNode node : nodes) {
            if (node instanceof FileSourceNode || node instanceof WebcamSourceNode) {
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
            // No source node - silently return (may be loading empty diagram)
            return;
        }

        // Get current frame from source
        Mat currentMat = null;
        if (sourceNode instanceof FileSourceNode) {
            currentMat = ((FileSourceNode) sourceNode).getLoadedImage();
        } else if (sourceNode instanceof WebcamSourceNode) {
            currentMat = ((WebcamSourceNode) sourceNode).getNextFrame();
        }

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

        // Find source node (FileSourceNode or WebcamSourceNode)
        PipelineNode sourceNode = null;
        for (PipelineNode node : nodes) {
            if (node instanceof FileSourceNode || node instanceof WebcamSourceNode) {
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

        // Validate source node
        if (sourceNode == null) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
            mb.setText("No Source");
            mb.setMessage("Please add a source node (Image Source or Webcam).");
            mb.open();
            return;
        }

        if (sourceNode instanceof FileSourceNode) {
            FileSourceNode isn = (FileSourceNode) sourceNode;
            if (isn.getLoadedImage() == null) {
                MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
                mb.setText("No Source");
                mb.setMessage("Please load an image in the source node first.");
                mb.open();
                return;
            }
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
        final PipelineNode finalSource = sourceNode;
        // Get FPS from source node
        double fps = 30.0;
        if (sourceNode instanceof FileSourceNode) {
            fps = ((FileSourceNode) sourceNode).getFps();
        } else if (sourceNode instanceof WebcamSourceNode) {
            fps = ((WebcamSourceNode) sourceNode).getFps();
        }
        final long frameDelayMs = (long) (1000.0 / fps);

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

                        if (node instanceof FileSourceNode) {
                            // File source node: get next frame (video or static image)
                            inputMat = ((FileSourceNode) node).getNextFrame();
                            if (inputMat == null) {
                                Thread.sleep(frameDelayMs);
                                continue;
                            }
                            outputMat = inputMat;
                        } else if (node instanceof WebcamSourceNode) {
                            // Webcam source node: get next frame from camera
                            inputMat = ((WebcamSourceNode) node).getNextFrame();
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
                        if (!display.isDisposed()) {
                            display.asyncExec(() -> {
                                if (canvas.isDisposed()) {
                                    thumbMat.release();
                                    return;
                                }
                                node.setOutputMat(thumbMat);
                                canvas.redraw();

                                // Update preview if this is the last node
                                if (outputQueue == null) {
                                    updatePreview(thumbMat);
                                }
                            });
                        } else {
                            thumbMat.release();
                        }

                        // Pass to next node
                        if (outputQueue != null) {
                            // Adaptive priority: adjust by 1 increment at a time
                            int queueSize = outputQueue.size();
                            System.out.println(node.getClass().getSimpleName() + " putting to queue, size=" + queueSize);

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
                            System.out.println(node.getClass().getSimpleName() + " put complete");
                        } else {
                            // Last node - cleanup
                            outputMat.release();
                        }

                        // Throttle source node based on video FPS
                        if (node instanceof FileSourceNode || node instanceof WebcamSourceNode) {
                            Thread.sleep(frameDelayMs);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (Exception e) {
                    System.err.println("Pipeline thread " + node.getClass().getSimpleName() + " exception:");
                    e.printStackTrace();
                }
                System.out.println("Pipeline thread " + node.getClass().getSimpleName() + " exited");
            });

            // Set priority based on node type
            if (node instanceof FileSourceNode || node instanceof WebcamSourceNode) {
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
        // Safety check: if no mouse button is pressed, reset drag state
        // This handles cases where mouseUp was missed (e.g., overlay capture)
        if ((e.stateMask & SWT.BUTTON_MASK) == 0) {
            if (isDragging) {
                isDragging = false;
                selectedNode = null;
            }
        }

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
                } else if (node instanceof FileSourceNode) {
                    ((FileSourceNode) node).showPropertiesDialog();
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

                // Edit Properties option (for ProcessingNode and FileSourceNode)
                if (node instanceof ProcessingNode) {
                    MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
                    editItem.setText("Edit Properties...");
                    editItem.addListener(SWT.Selection, evt -> {
                        ((ProcessingNode) node).showPropertiesDialog();
                    });

                    new MenuItem(contextMenu, SWT.SEPARATOR);
                } else if (node instanceof FileSourceNode) {
                    MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
                    editItem.setText("Edit Properties...");
                    editItem.addListener(SWT.Selection, evt -> {
                        ((FileSourceNode) node).showPropertiesDialog();
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
        addFileSourceNodeAt(50, 100);
        addEffectNodeAt("Grayscale", 300, 100);
        addEffectNodeAt("GaussianBlur", 500, 100);
        addEffectNodeAt("Threshold", 700, 100);

        if (nodes.size() >= 4) {
            connections.add(new Connection(nodes.get(0), nodes.get(1)));
            connections.add(new Connection(nodes.get(1), nodes.get(2)));
            connections.add(new Connection(nodes.get(2), nodes.get(3)));
        }
    }

    private void addFileSourceNode() {
        addFileSourceNodeAt(50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addFileSourceNodeAt(int x, int y) {
        FileSourceNode node = new FileSourceNode(shell, display, canvas, x, y);
        nodes.add(node);
        canvas.redraw();
    }

    private void addWebcamSourceNode() {
        addWebcamSourceNodeAt(50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addWebcamSourceNodeAt(int x, int y) {
        WebcamSourceNode node = new WebcamSourceNode(shell, display, canvas, x, y);
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
        // Normalize type name for backward compatibility
        String normalizedType = normalizeNodeType(type);

        // Try to create node using registry
        ProcessingNode node = NodeRegistry.createNode(normalizedType, display, shell, x, y);

        if (node != null) {
            return node;
        }

        // For any unknown type, create an UnknownNode as placeholder
        System.err.println("Unknown effect type: " + type + ", creating UnknownNode as placeholder");
        return new UnknownNode(display, shell, x, y, type);
    }

    // Normalize node type names for backward compatibility
    private String normalizeNodeType(String type) {
        switch (type) {
            // Blur
            case "Gaussian Blur":
            case "Blur":
                return "GaussianBlur";
            case "Median Blur":
                return "MedianBlur";
            case "Bilateral Filter":
                return "BilateralFilter";
            case "Mean Shift":
            case "Mean Shift Filter":
                return "MeanShift";

            // Basic
            case "Threshold (Simple)":
                return "Threshold";
            case "Color Convert":
                return "Grayscale";
            case "Adaptive Threshold":
            case "Threshold (Adaptive)":
                return "AdaptiveThreshold";
            case "CLAHE: Contrast Enhancement":
                return "CLAHE";
            case "Color In Range":
                return "ColorInRange";

            // Edge detection
            case "Canny Edge":
            case "Edges Canny":
                return "CannyEdge";
            case "Edges Laplacian":
                return "Laplacian";
            case "Gradient Sobel":
                return "Sobel";
            case "Edges Scharr":
                return "Scharr";

            // Morphological
            case "Morph Erode":
                return "Erode";
            case "Morph Dilate":
                return "Dilate";
            case "Morph Open":
                return "MorphOpen";
            case "Morph Close":
                return "MorphClose";

            // Detection
            case "Hough Circles":
                return "HoughCircles";
            case "Hough Lines":
                return "HoughLines";
            case "Harris Corners":
                return "HarrisCorners";
            case "Shi-Tomasi":
                return "ShiTomasi";
            case "Blob Detector":
                return "BlobDetector";
            case "ORB Features":
                return "ORBFeatures";
            case "SIFT Features":
                return "SIFTFeatures";
            case "Connected Components":
                return "ConnectedComponents";

            // Transform
            case "Warp Affine":
                return "WarpAffine";

            // Filter
            case "FFT High-Pass Filter":
                return "FFTHighPass";
            case "Bit Planes Grayscale":
                return "BitPlanesGrayscale";
            case "Bit Planes Color":
                return "BitPlanesColor";

            default:
                return type;
        }
    }
}
