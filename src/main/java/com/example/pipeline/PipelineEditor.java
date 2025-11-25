package com.example.pipeline;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;
import org.eclipse.swt.custom.SashForm;
import org.eclipse.swt.custom.ScrolledComposite;
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
        NodeRegistry.register("AdaptiveThreshold", "Basic", AdaptiveThresholdNode.class);
        NodeRegistry.register("BitPlanesColor", "Basic", BitPlanesColorNode.class);
        NodeRegistry.register("BitPlanesGrayscale", "Basic", BitPlanesGrayscaleNode.class);
        NodeRegistry.register("CLAHE Contrast", "Basic", CLAHENode.class);
        NodeRegistry.register("Gain", "Basic", GainNode.class);
        NodeRegistry.register("Invert", "Basic", InvertNode.class);
        NodeRegistry.register("Threshold", "Basic", ThresholdNode.class);

        // Blur nodes
        NodeRegistry.register("BilateralFilter", "Blur", BilateralFilterNode.class);
        NodeRegistry.register("BoxBlur", "Blur", BoxBlurNode.class);
        NodeRegistry.register("GaussianBlur", "Blur", GaussianBlurNode.class);
        NodeRegistry.register("MeanShift", "Blur", MeanShiftFilterNode.class);
        NodeRegistry.register("MedianBlur", "Blur", MedianBlurNode.class);

        // Content nodes (shapes and text)
        NodeRegistry.register("Arrow", "Content", ArrowNode.class);
        NodeRegistry.register("Circle", "Content", CircleNode.class);
        NodeRegistry.register("Ellipse", "Content", EllipseNode.class);
        NodeRegistry.register("Line", "Content", LineNode.class);
        NodeRegistry.register("Rectangle", "Content", RectangleNode.class);
        NodeRegistry.register("Text", "Content", TextNode.class);

        // Edge detection nodes
        NodeRegistry.register("CannyEdge", "Edge Detection", CannyEdgeNode.class);
        NodeRegistry.register("Laplacian", "Edge Detection", LaplacianNode.class);
        NodeRegistry.register("Sobel", "Edge Detection", SobelNode.class);
        NodeRegistry.register("Scharr", "Edge Detection", ScharrNode.class);

        // Filter nodes
        NodeRegistry.register("BitwiseNot", "Filter", BitwiseNotNode.class);
        NodeRegistry.register("ColorInRange", "Filter", ColorInRangeNode.class);
        NodeRegistry.register("FFTHighPass", "Filter", FFTHighPassFilterNode.class);
        NodeRegistry.register("FFTLowPass", "Filter", FFTLowPassFilterNode.class);
        NodeRegistry.register("Filter2D", "Filter", Filter2DNode.class);

        // Morphological nodes
        NodeRegistry.register("Dilate", "Morphological", DilateNode.class);
        NodeRegistry.register("Erode", "Morphological", ErodeNode.class);
        NodeRegistry.register("MorphClose", "Morphological", MorphCloseNode.class);
        NodeRegistry.register("MorphOpen", "Morphological", MorphOpenNode.class);
        NodeRegistry.register("MorphologyEx", "Morphological", MorphologyExNode.class);

        // Transform nodes
        NodeRegistry.register("Crop", "Transform", CropNode.class);
        NodeRegistry.register("WarpAffine", "Transform", WarpAffineNode.class);

        // Detection nodes
        NodeRegistry.register("BlobDetector", "Detection", BlobDetectorNode.class);
        NodeRegistry.register("ConnectedComponents", "Detection", ConnectedComponentsNode.class);
        NodeRegistry.register("Contours", "Detection", ContoursNode.class);
        NodeRegistry.register("HarrisCorners", "Detection", HarrisCornersNode.class);
        NodeRegistry.register("HoughCircles", "Detection", HoughCirclesNode.class);
        NodeRegistry.register("HoughLines", "Detection", HoughLinesNode.class);
        NodeRegistry.register("MatchTemplate", "Detection", MatchTemplateNode.class);
        NodeRegistry.register("ORBFeatures", "Detection", ORBFeaturesNode.class);
        NodeRegistry.register("ShiTomasi", "Detection", ShiTomasiCornersNode.class);
        NodeRegistry.register("SIFTFeatures", "Detection", SIFTFeaturesNode.class);

        // Dual Input nodes
        NodeRegistry.register("AddClamp", "Dual Input Nodes", AddClampNode.class);
        NodeRegistry.register("AddWeighted", "Dual Input Nodes", AddWeightedNode.class);
        NodeRegistry.register("BitwiseAnd", "Dual Input Nodes", BitwiseAndNode.class);
        NodeRegistry.register("BitwiseOr", "Dual Input Nodes", BitwiseOrNode.class);
        NodeRegistry.register("BitwiseXor", "Dual Input Nodes", BitwiseXorNode.class);
        NodeRegistry.register("SubtractClamp", "Dual Input Nodes", SubtractClampNode.class);

        // Visualization nodes
        NodeRegistry.register("Histogram", "Visualization", HistogramNode.class);

        // Register aliases for backward compatibility with renamed nodes
        NodeRegistry.registerAlias("Canny Edge", "CannyEdge");
        NodeRegistry.registerAlias("Canny Edges", "CannyEdge");
        NodeRegistry.registerAlias("Laplacian Edges", "Laplacian");
        NodeRegistry.registerAlias("Sobel Edges", "Sobel");
        NodeRegistry.registerAlias("Scharr Edges", "Scharr");
        NodeRegistry.registerAlias("Shi-Tomasi", "ShiTomasi");
        NodeRegistry.registerAlias("Shi-Tomasi Corners", "ShiTomasi");
        NodeRegistry.registerAlias("Harris Corners", "HarrisCorners");
        NodeRegistry.registerAlias("Hough Circles", "HoughCircles");
        NodeRegistry.registerAlias("Hough Lines", "HoughLines");
        NodeRegistry.registerAlias("Blob Detector", "BlobDetector");
        NodeRegistry.registerAlias("ORB Features", "ORBFeatures");
        NodeRegistry.registerAlias("SIFT Features", "SIFTFeatures");
        NodeRegistry.registerAlias("Connected Components", "ConnectedComponents");
        NodeRegistry.registerAlias("Color In Range", "ColorInRange");
        NodeRegistry.registerAlias("Bilateral Filter", "BilateralFilter");
        NodeRegistry.registerAlias("Box Blur", "BoxBlur");
        NodeRegistry.registerAlias("Gaussian Blur", "GaussianBlur");
        NodeRegistry.registerAlias("Median Blur", "MedianBlur");
        NodeRegistry.registerAlias("Mean Shift", "MeanShift");
        NodeRegistry.registerAlias("Mean Shift Blur", "MeanShift");
        NodeRegistry.registerAlias("Mean Shift Filter", "MeanShift");
        NodeRegistry.registerAlias("Unknown: Mean Shift Blur", "MeanShift");
        NodeRegistry.registerAlias("Grayscale/Color Convert", "Grayscale");
        NodeRegistry.registerAlias("Unknown: Grayscale/Color Convert", "Grayscale");
        NodeRegistry.registerAlias("Warp Affine", "WarpAffine");
        NodeRegistry.registerAlias("Match Template", "MatchTemplate");
        NodeRegistry.registerAlias("Unknown: Match Template", "MatchTemplate");
        NodeRegistry.registerAlias("Add Clamp", "AddClamp");
        NodeRegistry.registerAlias("Add Weighted", "AddWeighted");
        NodeRegistry.registerAlias("Subtract Clamp", "SubtractClamp");
        NodeRegistry.registerAlias("Bitwise And", "BitwiseAnd");
        NodeRegistry.registerAlias("Bitwise Or", "BitwiseOr");
        NodeRegistry.registerAlias("Bitwise Xor", "BitwiseXor");
        NodeRegistry.registerAlias("Bitwise NOT", "BitwiseNot");
        NodeRegistry.registerAlias("BitwiseNOT", "BitwiseNot");
        NodeRegistry.registerAlias("Filter 2D", "Filter2D");
        NodeRegistry.registerAlias("FFT Filter", "FFTHighPass");
        NodeRegistry.registerAlias("FFT High-Pass Filter", "FFTHighPass");
        NodeRegistry.registerAlias("Unknown: FFT High-Pass Filter", "FFTHighPass");
        NodeRegistry.registerAlias("FFTHighPassFilter", "FFTHighPass");
        NodeRegistry.registerAlias("FFT Low-Pass Filter", "FFTLowPass");
        NodeRegistry.registerAlias("Unknown: FFT Low-Pass Filter", "FFTLowPass");
        NodeRegistry.registerAlias("FFTLowPassFilter", "FFTLowPass");
        NodeRegistry.registerAlias("Morphology Ex", "MorphologyEx");
    }

    private Shell shell;
    private Display display;
    private Canvas canvas;
    private ScrolledComposite scrolledCanvas;
    private Canvas previewCanvas;
    private SashForm sashForm;
    private Image previewImage;
    private Label statusBar;
    private Label nodeCountLabel;
    private Combo zoomCombo;
    private double zoomLevel = 1.0; // 1.0 = 100%
    private static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200, 300, 400};

    // Debounce for node button double-clicks
    private static final int NODE_BUTTON_DEBOUNCE_MS = 300;
    private long lastNodeButtonClickTime = 0;

    // Convert screen coordinates to canvas coordinates (accounting for zoom)
    private int toCanvasX(int screenX) {
        return (int) (screenX / zoomLevel);
    }
    private int toCanvasY(int screenY) {
        return (int) (screenY / zoomLevel);
    }
    private Point toCanvasPoint(int screenX, int screenY) {
        return new Point(toCanvasX(screenX), toCanvasY(screenY));
    }

    private List<PipelineNode> nodes = new ArrayList<>();
    private List<Connection> connections = new ArrayList<>();

    private PipelineNode selectedNode = null;
    private Point dragOffset = null;
    private boolean isDragging = false;
    private boolean nodesMoved = false;

    // Connection drawing state
    private PipelineNode connectionSource = null;
    private Point connectionEndPoint = null;

    // Dangling connections (one end attached, one end free)
    private java.util.List<DanglingConnection> danglingConnections = new java.util.ArrayList<>();
    private java.util.List<ReverseDanglingConnection> reverseDanglingConnections = new java.util.ArrayList<>();
    private java.util.List<FreeConnection> freeConnections = new java.util.ArrayList<>();
    private PipelineNode connectionTarget = null; // For reverse dragging (from target end)
    private int targetInputIndex = 1; // Which input to target (1 or 2 for dual-input nodes)

    // Free connection dragging state
    private Point freeConnectionFixedEnd = null; // The end that stays fixed while dragging the other
    private boolean draggingFreeConnectionSource = false; // true if dragging source end, false if dragging target end

    // Selection state
    private Set<PipelineNode> selectedNodes = new HashSet<>();
    private Set<Connection> selectedConnections = new HashSet<>();

    // Search/filter state for toolbar
    private Text searchBox;
    private java.util.List<SearchableButton> searchableButtons = new java.util.ArrayList<>();
    private Composite toolbarContent;
    private org.eclipse.swt.custom.ScrolledComposite scrolledToolbar;
    private int selectedButtonIndex = -1; // -1 means no selection, use first visible

    // Inner class to track button metadata for search
    private static class SearchableButton {
        Button button;
        Label categoryLabel; // The category label above this button (if first in category)
        Label separator;     // The separator above the category label
        String nodeName;
        String category;
        Runnable action;
        boolean isFirstInCategory;

        SearchableButton(Button button, String nodeName, String category, Runnable action) {
            this.button = button;
            this.nodeName = nodeName;
            this.category = category;
            this.action = action;
        }
    }
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
    private Menu openRecentMenu;

    // Threading state
    private AtomicBoolean pipelineRunning = new AtomicBoolean(false);
    private Button startStopBtn;

    public static void main(String[] args) {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();

        PipelineEditor editor = new PipelineEditor();
        editor.run();
    }

    private String currentFilePath = null;
    private boolean isDirty = false; // Track unsaved changes

    private void markDirty() {
        isDirty = true;
    }

    private void clearDirty() {
        isDirty = false;
    }

    /**
     * Check if there are unsaved changes and prompt user.
     * Returns true if OK to proceed (saved, discarded, or no changes).
     * Returns false if user cancelled.
     */
    private boolean checkUnsavedChanges() {
        if (!isDirty) {
            return true;
        }

        MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.YES | SWT.NO | SWT.CANCEL);
        mb.setText("Unsaved Changes");
        mb.setMessage("You have unsaved changes. Do you want to save before continuing?");
        int result = mb.open();

        if (result == SWT.YES) {
            // Save the file
            if (currentFilePath != null) {
                saveDiagramToPath(currentFilePath);
            } else {
                FileDialog dialog = new FileDialog(shell, SWT.SAVE);
                dialog.setText("Save Pipeline");
                dialog.setFilterExtensions(new String[]{"*.json"});
                dialog.setFilterNames(new String[]{"Pipeline Files (*.json)"});
                String path = dialog.open();
                if (path != null) {
                    if (!path.toLowerCase().endsWith(".json")) {
                        path += ".json";
                    }
                    saveDiagramToPath(path);
                } else {
                    return false; // User cancelled save dialog
                }
            }
            return true;
        } else if (result == SWT.NO) {
            return true; // Discard changes
        } else {
            return false; // Cancel
        }
    }

    public void run() {
        // Set application name for macOS menu bar (must be before Display creation)
        Display.setAppName("OpenCV");
        display = new Display();

        shell = new Shell(display);
        shell.setText("OpenCV Pipeline Editor - (untitled)");
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

        // Add close listener to check for unsaved changes
        shell.addListener(SWT.Close, event -> {
            event.doit = checkUnsavedChanges();
        });

        shell.open();

        // Set focus to canvas to avoid search box focus ring on startup
        canvas.setFocus();

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
        // Check for unsaved changes
        if (!checkUnsavedChanges()) {
            return;
        }

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
                    updateOpenRecentMenu();
                }
            });
        }
    }


    private void loadDiagramFromPath(String path) {
        try {
            // Clear existing
            for (PipelineNode node : nodes) {
                if (node instanceof SourceNode) {
                    ((SourceNode) node).dispose();
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
                    if (nodeObj.has("threadPriority")) {
                        node.setThreadPriority(nodeObj.get("threadPriority").getAsInt());
                    }
                    if (nodeObj.has("workUnitsCompleted")) {
                        node.setWorkUnitsCompleted(nodeObj.get("workUnitsCompleted").getAsLong());
                    }
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
                    if (nodeObj.has("threadPriority")) {
                        node.setThreadPriority(nodeObj.get("threadPriority").getAsInt());
                    }
                    if (nodeObj.has("workUnitsCompleted")) {
                        node.setWorkUnitsCompleted(nodeObj.get("workUnitsCompleted").getAsLong());
                    }
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
                    // Initialize camera after all properties are loaded
                    node.initAfterLoad();
                    nodes.add(node);
                } else if ("BlankSource".equals(type)) {
                    BlankSourceNode node = new BlankSourceNode(shell, display, x, y);
                    if (nodeObj.has("threadPriority")) {
                        node.setThreadPriority(nodeObj.get("threadPriority").getAsInt());
                    }
                    if (nodeObj.has("workUnitsCompleted")) {
                        node.setWorkUnitsCompleted(nodeObj.get("workUnitsCompleted").getAsLong());
                    }
                    if (nodeObj.has("imageWidth")) {
                        node.setImageWidth(nodeObj.get("imageWidth").getAsInt());
                    }
                    if (nodeObj.has("imageHeight")) {
                        node.setImageHeight(nodeObj.get("imageHeight").getAsInt());
                    }
                    if (nodeObj.has("colorIndex")) {
                        node.setColorIndex(nodeObj.get("colorIndex").getAsInt());
                    }
                    if (nodeObj.has("fpsIndex")) {
                        node.setFpsIndex(nodeObj.get("fpsIndex").getAsInt());
                    }
                    nodes.add(node);
                } else if ("Processing".equals(type)) {
                    String name = nodeObj.get("name").getAsString();
                    ProcessingNode node = createEffectNode(name, x, y);
                    if (node != null) {
                        // Load thread priority
                        if (nodeObj.has("threadPriority")) {
                            node.setThreadPriority(nodeObj.get("threadPriority").getAsInt());
                        }
                        if (nodeObj.has("workUnitsCompleted")) {
                            node.setWorkUnitsCompleted(nodeObj.get("workUnitsCompleted").getAsLong());
                        }
                        // Load node-specific properties
                        if (node instanceof GaussianBlurNode) {
                            GaussianBlurNode gbn = (GaussianBlurNode) node;
                            if (nodeObj.has("kernelSizeX")) gbn.setKernelSizeX(nodeObj.get("kernelSizeX").getAsInt());
                            if (nodeObj.has("kernelSizeY")) gbn.setKernelSizeY(nodeObj.get("kernelSizeY").getAsInt());
                            if (nodeObj.has("sigmaX")) gbn.setSigmaX(nodeObj.get("sigmaX").getAsDouble());
                        } else if (node instanceof BoxBlurNode) {
                            BoxBlurNode bbn = (BoxBlurNode) node;
                            if (nodeObj.has("kernelSizeX")) bbn.setKernelSizeX(nodeObj.get("kernelSizeX").getAsInt());
                            if (nodeObj.has("kernelSizeY")) bbn.setKernelSizeY(nodeObj.get("kernelSizeY").getAsInt());
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
                            if (nodeObj.has("returnedThreshold")) tn.setReturnedThreshold(nodeObj.get("returnedThreshold").getAsDouble());
                        } else if (node instanceof ContoursNode) {
                            ContoursNode cn = (ContoursNode) node;
                            if (nodeObj.has("thresholdValue")) cn.setThresholdValue(nodeObj.get("thresholdValue").getAsInt());
                            if (nodeObj.has("retrievalMode")) cn.setRetrievalMode(nodeObj.get("retrievalMode").getAsInt());
                            if (nodeObj.has("approxMethod")) cn.setApproxMethod(nodeObj.get("approxMethod").getAsInt());
                            if (nodeObj.has("thickness")) cn.setThickness(nodeObj.get("thickness").getAsInt());
                            if (nodeObj.has("colorR")) cn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) cn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) cn.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("showOriginal")) cn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                            if (nodeObj.has("sortMethod")) cn.setSortMethod(nodeObj.get("sortMethod").getAsInt());
                            if (nodeObj.has("minIndex")) cn.setMinIndex(nodeObj.get("minIndex").getAsInt());
                            if (nodeObj.has("maxIndex")) cn.setMaxIndex(nodeObj.get("maxIndex").getAsInt());
                            if (nodeObj.has("drawMode")) cn.setDrawMode(nodeObj.get("drawMode").getAsInt());
                        } else if (node instanceof BlobDetectorNode) {
                            BlobDetectorNode bdn = (BlobDetectorNode) node;
                            if (nodeObj.has("minThreshold")) bdn.setMinThreshold(nodeObj.get("minThreshold").getAsInt());
                            if (nodeObj.has("maxThreshold")) bdn.setMaxThreshold(nodeObj.get("maxThreshold").getAsInt());
                            if (nodeObj.has("showOriginal")) bdn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                            if (nodeObj.has("filterByArea")) bdn.setFilterByArea(nodeObj.get("filterByArea").getAsBoolean());
                            if (nodeObj.has("minArea")) bdn.setMinArea(nodeObj.get("minArea").getAsInt());
                            if (nodeObj.has("maxArea")) bdn.setMaxArea(nodeObj.get("maxArea").getAsInt());
                            if (nodeObj.has("filterByCircularity")) bdn.setFilterByCircularity(nodeObj.get("filterByCircularity").getAsBoolean());
                            if (nodeObj.has("minCircularity")) bdn.setMinCircularity(nodeObj.get("minCircularity").getAsInt());
                            if (nodeObj.has("filterByConvexity")) bdn.setFilterByConvexity(nodeObj.get("filterByConvexity").getAsBoolean());
                            if (nodeObj.has("minConvexity")) bdn.setMinConvexity(nodeObj.get("minConvexity").getAsInt());
                            if (nodeObj.has("filterByInertia")) bdn.setFilterByInertia(nodeObj.get("filterByInertia").getAsBoolean());
                            if (nodeObj.has("minInertiaRatio")) bdn.setMinInertiaRatio(nodeObj.get("minInertiaRatio").getAsInt());
                            if (nodeObj.has("filterByColor")) bdn.setFilterByColor(nodeObj.get("filterByColor").getAsBoolean());
                            if (nodeObj.has("blobColor")) bdn.setBlobColor(nodeObj.get("blobColor").getAsInt());
                            if (nodeObj.has("colorR")) bdn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) bdn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) bdn.setColorB(nodeObj.get("colorB").getAsInt());
                        } else if (node instanceof HoughCirclesNode) {
                            HoughCirclesNode hcn = (HoughCirclesNode) node;
                            if (nodeObj.has("showOriginal")) hcn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                            if (nodeObj.has("minDist")) hcn.setMinDist(nodeObj.get("minDist").getAsInt());
                            if (nodeObj.has("param1")) hcn.setParam1(nodeObj.get("param1").getAsInt());
                            if (nodeObj.has("param2")) hcn.setParam2(nodeObj.get("param2").getAsInt());
                            if (nodeObj.has("minRadius")) hcn.setMinRadius(nodeObj.get("minRadius").getAsInt());
                            if (nodeObj.has("maxRadius")) hcn.setMaxRadius(nodeObj.get("maxRadius").getAsInt());
                            if (nodeObj.has("thickness")) hcn.setThickness(nodeObj.get("thickness").getAsInt());
                            if (nodeObj.has("drawCenter")) hcn.setDrawCenter(nodeObj.get("drawCenter").getAsBoolean());
                            if (nodeObj.has("colorR")) hcn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) hcn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) hcn.setColorB(nodeObj.get("colorB").getAsInt());
                        } else if (node instanceof HarrisCornersNode) {
                            HarrisCornersNode harn = (HarrisCornersNode) node;
                            if (nodeObj.has("showOriginal")) harn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                            if (nodeObj.has("drawFeatures")) harn.setDrawFeatures(nodeObj.get("drawFeatures").getAsBoolean());
                            if (nodeObj.has("blockSize")) harn.setBlockSize(nodeObj.get("blockSize").getAsInt());
                            if (nodeObj.has("ksize")) harn.setKsize(nodeObj.get("ksize").getAsInt());
                            if (nodeObj.has("kPercent")) harn.setKPercent(nodeObj.get("kPercent").getAsInt());
                            if (nodeObj.has("thresholdPercent")) harn.setThresholdPercent(nodeObj.get("thresholdPercent").getAsInt());
                            if (nodeObj.has("markerSize")) harn.setMarkerSize(nodeObj.get("markerSize").getAsInt());
                            if (nodeObj.has("colorR")) harn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) harn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) harn.setColorB(nodeObj.get("colorB").getAsInt());
                        } else if (node instanceof ShiTomasiCornersNode) {
                            ShiTomasiCornersNode stc = (ShiTomasiCornersNode) node;
                            if (nodeObj.has("maxCorners")) stc.setMaxCorners(nodeObj.get("maxCorners").getAsInt());
                            if (nodeObj.has("qualityLevel")) stc.setQualityLevel(nodeObj.get("qualityLevel").getAsInt());
                            if (nodeObj.has("minDistance")) stc.setMinDistance(nodeObj.get("minDistance").getAsInt());
                            if (nodeObj.has("blockSize")) stc.setBlockSize(nodeObj.get("blockSize").getAsInt());
                            if (nodeObj.has("useHarrisDetector")) stc.setUseHarrisDetector(nodeObj.get("useHarrisDetector").getAsBoolean());
                            if (nodeObj.has("kPercent")) stc.setKPercent(nodeObj.get("kPercent").getAsInt());
                            if (nodeObj.has("markerSize")) stc.setMarkerSize(nodeObj.get("markerSize").getAsInt());
                            if (nodeObj.has("colorR")) stc.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) stc.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) stc.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("drawFeatures")) stc.setDrawFeatures(nodeObj.get("drawFeatures").getAsBoolean());
                        } else if (node instanceof ScharrNode) {
                            ScharrNode sn = (ScharrNode) node;
                            if (nodeObj.has("directionIndex")) sn.setDirectionIndex(nodeObj.get("directionIndex").getAsInt());
                            if (nodeObj.has("scalePercent")) sn.setScalePercent(nodeObj.get("scalePercent").getAsInt());
                            if (nodeObj.has("delta")) sn.setDelta(nodeObj.get("delta").getAsInt());
                        } else if (node instanceof SobelNode) {
                            SobelNode sn = (SobelNode) node;
                            if (nodeObj.has("dx")) sn.setDx(nodeObj.get("dx").getAsInt());
                            if (nodeObj.has("dy")) sn.setDy(nodeObj.get("dy").getAsInt());
                            if (nodeObj.has("kernelSizeIndex")) sn.setKernelSizeIndex(nodeObj.get("kernelSizeIndex").getAsInt());
                        } else if (node instanceof LaplacianNode) {
                            LaplacianNode ln = (LaplacianNode) node;
                            if (nodeObj.has("kernelSizeIndex")) ln.setKernelSizeIndex(nodeObj.get("kernelSizeIndex").getAsInt());
                            if (nodeObj.has("scalePercent")) ln.setScalePercent(nodeObj.get("scalePercent").getAsInt());
                            if (nodeObj.has("delta")) ln.setDelta(nodeObj.get("delta").getAsInt());
                            if (nodeObj.has("useAbsolute")) ln.setUseAbsolute(nodeObj.get("useAbsolute").getAsBoolean());
                        } else if (node instanceof GainNode) {
                            GainNode gn = (GainNode) node;
                            if (nodeObj.has("gain")) gn.setGain(nodeObj.get("gain").getAsDouble());
                        } else if (node instanceof Filter2DNode) {
                            Filter2DNode f2d = (Filter2DNode) node;
                            if (nodeObj.has("kernelSize")) f2d.setKernelSize(nodeObj.get("kernelSize").getAsInt());
                            if (nodeObj.has("kernelValues")) {
                                JsonArray arr = nodeObj.getAsJsonArray("kernelValues");
                                int[] values = new int[arr.size()];
                                for (int j = 0; j < arr.size(); j++) {
                                    values[j] = arr.get(j).getAsInt();
                                }
                                f2d.setKernelValues(values);
                            }
                        } else if (node instanceof MorphologyExNode) {
                            MorphologyExNode men = (MorphologyExNode) node;
                            if (nodeObj.has("operationIndex")) men.setOperationIndex(nodeObj.get("operationIndex").getAsInt());
                            if (nodeObj.has("shapeIndex")) men.setShapeIndex(nodeObj.get("shapeIndex").getAsInt());
                            if (nodeObj.has("kernelWidth")) men.setKernelWidth(nodeObj.get("kernelWidth").getAsInt());
                            if (nodeObj.has("kernelHeight")) men.setKernelHeight(nodeObj.get("kernelHeight").getAsInt());
                            if (nodeObj.has("iterations")) men.setIterations(nodeObj.get("iterations").getAsInt());
                            if (nodeObj.has("anchorX")) men.setAnchorX(nodeObj.get("anchorX").getAsInt());
                            if (nodeObj.has("anchorY")) men.setAnchorY(nodeObj.get("anchorY").getAsInt());
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
                        } else if (node instanceof RectangleNode) {
                            RectangleNode rn = (RectangleNode) node;
                            if (nodeObj.has("x1")) rn.setX1(nodeObj.get("x1").getAsInt());
                            if (nodeObj.has("y1")) rn.setY1(nodeObj.get("y1").getAsInt());
                            if (nodeObj.has("x2")) rn.setX2(nodeObj.get("x2").getAsInt());
                            if (nodeObj.has("y2")) rn.setY2(nodeObj.get("y2").getAsInt());
                            if (nodeObj.has("colorR")) rn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) rn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) rn.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("thickness")) rn.setThickness(nodeObj.get("thickness").getAsInt());
                            if (nodeObj.has("filled")) rn.setFilled(nodeObj.get("filled").getAsBoolean());
                        } else if (node instanceof CircleNode) {
                            CircleNode cn = (CircleNode) node;
                            if (nodeObj.has("centerX")) cn.setCenterX(nodeObj.get("centerX").getAsInt());
                            if (nodeObj.has("centerY")) cn.setCenterY(nodeObj.get("centerY").getAsInt());
                            if (nodeObj.has("radius")) cn.setRadius(nodeObj.get("radius").getAsInt());
                            if (nodeObj.has("colorR")) cn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) cn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) cn.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("thickness")) cn.setThickness(nodeObj.get("thickness").getAsInt());
                            if (nodeObj.has("filled")) cn.setFilled(nodeObj.get("filled").getAsBoolean());
                        } else if (node instanceof EllipseNode) {
                            EllipseNode en = (EllipseNode) node;
                            if (nodeObj.has("centerX")) en.setCenterX(nodeObj.get("centerX").getAsInt());
                            if (nodeObj.has("centerY")) en.setCenterY(nodeObj.get("centerY").getAsInt());
                            if (nodeObj.has("axisX")) en.setAxisX(nodeObj.get("axisX").getAsInt());
                            if (nodeObj.has("axisY")) en.setAxisY(nodeObj.get("axisY").getAsInt());
                            if (nodeObj.has("angle")) en.setAngle(nodeObj.get("angle").getAsInt());
                            if (nodeObj.has("startAngle")) en.setStartAngle(nodeObj.get("startAngle").getAsInt());
                            if (nodeObj.has("endAngle")) en.setEndAngle(nodeObj.get("endAngle").getAsInt());
                            if (nodeObj.has("colorR")) en.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) en.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) en.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("thickness")) en.setThickness(nodeObj.get("thickness").getAsInt());
                            if (nodeObj.has("filled")) en.setFilled(nodeObj.get("filled").getAsBoolean());
                        } else if (node instanceof LineNode) {
                            LineNode ln = (LineNode) node;
                            if (nodeObj.has("x1")) ln.setX1(nodeObj.get("x1").getAsInt());
                            if (nodeObj.has("y1")) ln.setY1(nodeObj.get("y1").getAsInt());
                            if (nodeObj.has("x2")) ln.setX2(nodeObj.get("x2").getAsInt());
                            if (nodeObj.has("y2")) ln.setY2(nodeObj.get("y2").getAsInt());
                            if (nodeObj.has("colorR")) ln.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) ln.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) ln.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("thickness")) ln.setThickness(nodeObj.get("thickness").getAsInt());
                        } else if (node instanceof ArrowNode) {
                            ArrowNode an = (ArrowNode) node;
                            if (nodeObj.has("x1")) an.setX1(nodeObj.get("x1").getAsInt());
                            if (nodeObj.has("y1")) an.setY1(nodeObj.get("y1").getAsInt());
                            if (nodeObj.has("x2")) an.setX2(nodeObj.get("x2").getAsInt());
                            if (nodeObj.has("y2")) an.setY2(nodeObj.get("y2").getAsInt());
                            if (nodeObj.has("tipLength")) an.setTipLength(nodeObj.get("tipLength").getAsDouble());
                            if (nodeObj.has("colorR")) an.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) an.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) an.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("thickness")) an.setThickness(nodeObj.get("thickness").getAsInt());
                        } else if (node instanceof CropNode) {
                            CropNode crn = (CropNode) node;
                            if (nodeObj.has("cropX")) crn.setCropX(nodeObj.get("cropX").getAsInt());
                            if (nodeObj.has("cropY")) crn.setCropY(nodeObj.get("cropY").getAsInt());
                            if (nodeObj.has("cropWidth")) crn.setCropWidth(nodeObj.get("cropWidth").getAsInt());
                            if (nodeObj.has("cropHeight")) crn.setCropHeight(nodeObj.get("cropHeight").getAsInt());
                        // AddClampNode and SubtractClampNode have no properties to load
                        } else if (node instanceof AddWeightedNode) {
                            AddWeightedNode awn = (AddWeightedNode) node;
                            if (nodeObj.has("alpha")) awn.setAlpha(nodeObj.get("alpha").getAsDouble());
                            if (nodeObj.has("beta")) awn.setBeta(nodeObj.get("beta").getAsDouble());
                            if (nodeObj.has("gamma")) awn.setGamma(nodeObj.get("gamma").getAsDouble());
                        } else if (node instanceof TextNode) {
                            TextNode tn = (TextNode) node;
                            if (nodeObj.has("text")) tn.setText(nodeObj.get("text").getAsString());
                            if (nodeObj.has("posX")) tn.setPosX(nodeObj.get("posX").getAsInt());
                            if (nodeObj.has("posY")) tn.setPosY(nodeObj.get("posY").getAsInt());
                            if (nodeObj.has("fontIndex")) tn.setFontIndex(nodeObj.get("fontIndex").getAsInt());
                            if (nodeObj.has("fontScale")) tn.setFontScale(nodeObj.get("fontScale").getAsDouble());
                            if (nodeObj.has("colorR")) tn.setColorR(nodeObj.get("colorR").getAsInt());
                            if (nodeObj.has("colorG")) tn.setColorG(nodeObj.get("colorG").getAsInt());
                            if (nodeObj.has("colorB")) tn.setColorB(nodeObj.get("colorB").getAsInt());
                            if (nodeObj.has("thickness")) tn.setThickness(nodeObj.get("thickness").getAsInt());
                            if (nodeObj.has("bold")) tn.setBold(nodeObj.get("bold").getAsBoolean());
                            if (nodeObj.has("italic")) tn.setItalic(nodeObj.get("italic").getAsBoolean());
                        } else if (node instanceof HistogramNode) {
                            HistogramNode hn = (HistogramNode) node;
                            if (nodeObj.has("modeIndex")) hn.setModeIndex(nodeObj.get("modeIndex").getAsInt());
                            if (nodeObj.has("backgroundMode")) hn.setBackgroundMode(nodeObj.get("backgroundMode").getAsInt());
                            if (nodeObj.has("fillBars")) hn.setFillBars(nodeObj.get("fillBars").getAsBoolean());
                            if (nodeObj.has("lineThickness")) hn.setLineThickness(nodeObj.get("lineThickness").getAsInt());
                            if (nodeObj.has("queuesInSync")) hn.setQueuesInSync(nodeObj.get("queuesInSync").getAsBoolean());
                        } else if (node instanceof MatchTemplateNode) {
                            MatchTemplateNode mtn = (MatchTemplateNode) node;
                            if (nodeObj.has("method")) mtn.setMethod(nodeObj.get("method").getAsInt());
                            if (nodeObj.has("queuesInSync")) mtn.setQueuesInSync(nodeObj.get("queuesInSync").getAsBoolean());
                            // Handle old showRectangle boolean (convert to outputMode)
                            if (nodeObj.has("showRectangle") && !nodeObj.has("outputMode")) {
                                mtn.setOutputMode(nodeObj.get("showRectangle").getAsBoolean() ? 1 : 0);
                            }
                            if (nodeObj.has("outputMode")) mtn.setOutputMode(nodeObj.get("outputMode").getAsInt());
                            if (nodeObj.has("rectColorR")) mtn.setRectColorR(nodeObj.get("rectColorR").getAsInt());
                            if (nodeObj.has("rectColorG")) mtn.setRectColorG(nodeObj.get("rectColorG").getAsInt());
                            if (nodeObj.has("rectColorB")) mtn.setRectColorB(nodeObj.get("rectColorB").getAsInt());
                            if (nodeObj.has("rectThickness")) mtn.setRectThickness(nodeObj.get("rectThickness").getAsInt());
                        }
                        // InvertNode has no properties to load
                        node.setOnChanged(() -> { markDirty(); executePipeline(); });
                        nodes.add(node);
                    }
                }
            }

            // Load thumbnails from cache for ProcessingNodes and WebcamSourceNodes
            String cacheDir = getCacheDir(path);
            for (int i = 0; i < nodes.size(); i++) {
                PipelineNode node = nodes.get(i);
                if (node instanceof WebcamSourceNode) {
                    ((WebcamSourceNode) node).loadThumbnailFromCache(cacheDir, i);
                } else if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).loadThumbnailFromCache(cacheDir, i);
                }
            }

            // Load connections
            JsonArray connsArray = root.getAsJsonArray("connections");
            for (JsonElement elem : connsArray) {
                JsonObject connObj = elem.getAsJsonObject();
                int sourceId = connObj.get("sourceId").getAsInt();
                int targetId = connObj.get("targetId").getAsInt();
                int inputIdx = connObj.has("inputIndex") ? connObj.get("inputIndex").getAsInt() : 1;
                if (sourceId >= 0 && sourceId < nodes.size() &&
                    targetId >= 0 && targetId < nodes.size()) {
                    Connection conn = new Connection(nodes.get(sourceId), nodes.get(targetId), inputIdx);
                    if (connObj.has("queueCapacity")) {
                        conn.setConfiguredCapacity(connObj.get("queueCapacity").getAsInt());
                    }
                    if (connObj.has("queueCount")) {
                        conn.setLastQueueSize(connObj.get("queueCount").getAsInt());
                    }
                    connections.add(conn);
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
                        if (dangObj.has("queueCapacity")) {
                            dc.setConfiguredCapacity(dangObj.get("queueCapacity").getAsInt());
                        }
                        if (dangObj.has("queueCount")) {
                            dc.setLastQueueSize(dangObj.get("queueCount").getAsInt());
                        }
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
                        if (revObj.has("queueCapacity")) {
                            rdc.setConfiguredCapacity(revObj.get("queueCapacity").getAsInt());
                        }
                        if (revObj.has("queueCount")) {
                            rdc.setLastQueueSize(revObj.get("queueCount").getAsInt());
                        }
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
                    if (freeObj.has("queueCapacity")) {
                        fc.setConfiguredCapacity(freeObj.get("queueCapacity").getAsInt());
                    }
                    if (freeObj.has("queueCount")) {
                        fc.setLastQueueSize(freeObj.get("queueCount").getAsInt());
                    }
                    freeConnections.add(fc);
                }
            }

            currentFilePath = path;
            addToRecentFiles(path);

            canvas.redraw();
            shell.setText("OpenCV Pipeline Editor - " + new File(path).getName());
            clearDirty();
            updateCanvasSize();

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
            if (node instanceof SourceNode) {
                ((SourceNode) node).dispose();
            }
        }
        nodes.clear();
        connections.clear();
        danglingConnections.clear();
        reverseDanglingConnections.clear();
        freeConnections.clear();
        currentFilePath = null;
        shell.setText("OpenCV Pipeline Editor - (untitled)");
        canvas.redraw();
    }

    private void createToolbar() {
        // Create outer container for search box + scrollable toolbar
        Composite toolbarContainer = new Composite(shell, SWT.BORDER);
        toolbarContainer.setLayoutData(new GridData(SWT.FILL, SWT.FILL, false, true));
        GridLayout containerLayout = new GridLayout(1, false);
        containerLayout.marginWidth = 0;
        containerLayout.marginHeight = 0;
        containerLayout.verticalSpacing = 0;
        toolbarContainer.setLayout(containerLayout);

        // Search label and box at top
        Label searchLabel = new Label(toolbarContainer, SWT.NONE);
        searchLabel.setText("Type to Search:");
        Font searchFont = new Font(display, "Arial", 13, SWT.BOLD);
        searchLabel.setFont(searchFont);
        searchLabel.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        searchBox = new Text(toolbarContainer, SWT.BORDER | SWT.SEARCH | SWT.ICON_SEARCH | SWT.ICON_CANCEL);
        searchBox.setMessage("Search nodes...");
        searchBox.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        searchBox.addModifyListener(e -> {
            clearButtonHighlighting();
            filterToolbarButtons();
        });
        // Handle cancel button (X) click
        searchBox.addListener(SWT.DefaultSelection, e -> {
            if (searchBox.getText().isEmpty()) {
                // X was clicked - already empty, nothing to do
            }
        });
        searchBox.addListener(SWT.KeyDown, e -> {
            if (e.keyCode == SWT.ESC) {
                searchBox.setText("");
            }
        });
        // The ICON_CANCEL should clear on click - need to use a traverse listener
        searchBox.addListener(SWT.MouseDown, e -> {
            // Check if click is in the cancel icon area (right side)
            org.eclipse.swt.graphics.Rectangle bounds = searchBox.getBounds();
            if (e.x > bounds.width - 20) {
                searchBox.setText("");
            }
        });
        searchBox.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.keyCode == SWT.CR || e.keyCode == SWT.KEYPAD_CR) {
                    // Enter pressed - add selected or first visible button's node
                    addSelectedNode();
                } else if (e.keyCode == SWT.ARROW_DOWN) {
                    navigateSelection(1);
                    e.doit = false; // Prevent cursor movement
                } else if (e.keyCode == SWT.ARROW_UP) {
                    navigateSelection(-1);
                    e.doit = false; // Prevent cursor movement
                }
            }
        });

        // Create scrollable container for the toolbar
        scrolledToolbar = new org.eclipse.swt.custom.ScrolledComposite(toolbarContainer, SWT.V_SCROLL);
        scrolledToolbar.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        scrolledToolbar.setExpandHorizontal(true);
        scrolledToolbar.setExpandVertical(true);

        toolbarContent = new Composite(scrolledToolbar, SWT.NONE);
        GridLayout toolbarLayout = new GridLayout(1, false);
        toolbarLayout.verticalSpacing = 0;  // No spacing between buttons
        toolbarLayout.marginHeight = 5;     // Reduce top/bottom margins

        // mbennett
        //toolbarLayout.marginWidth = 5;      // Keep side margins reasonable
        toolbarLayout.marginWidth = 12;       // 15;

        toolbarContent.setLayout(toolbarLayout);

        // Set darker green background for toolbar (better text visibility in dark mode)
        Color toolbarGreen = new Color(160, 200, 160);
        toolbarContent.setBackground(toolbarGreen);
        scrolledToolbar.setBackground(toolbarGreen);
        toolbarContainer.setBackground(toolbarGreen);

        Font boldFont = new Font(display, "Arial", 13, SWT.BOLD);

        // Inputs section (not from registry)
        Label inputsLabel = new Label(toolbarContent, SWT.NONE);
        inputsLabel.setText("Sources:");
        inputsLabel.setFont(boldFont);

        createSearchableButton(toolbarContent, "File Source", "Inputs", () -> addFileSourceNode(), inputsLabel, null, true);
        createSearchableButton(toolbarContent, "Webcam Source", "Inputs", () -> addWebcamSourceNode(), null, null, false);
        createSearchableButton(toolbarContent, "Blank Source", "Inputs", () -> addBlankSourceNode(), null, null, false);

        // Generate buttons from NodeRegistry grouped by category
        // Get display names from temp node instances
        java.util.Map<String, java.util.List<String[]>> categoryNodes = new java.util.LinkedHashMap<>();

        for (com.example.pipeline.registry.NodeRegistry.NodeInfo info :
             com.example.pipeline.registry.NodeRegistry.getAllNodes()) {
            // Create temp node to get display name and category
            ProcessingNode tempNode = com.example.pipeline.registry.NodeRegistry.createNode(
                info.name, display, shell, 0, 0);
            if (tempNode != null) {
                String displayName = tempNode.getDisplayName();
                String category = tempNode.getCategory();
                tempNode.disposeThumbnail();

                categoryNodes.computeIfAbsent(category, k -> new java.util.ArrayList<>())
                    .add(new String[]{displayName, info.name});
            }
        }

        // Create buttons for each category
        for (java.util.Map.Entry<String, java.util.List<String[]>> entry : categoryNodes.entrySet()) {
            String category = entry.getKey();
            java.util.List<String[]> nodeList = entry.getValue();

            // Separator
            Label separator = new Label(toolbarContent, SWT.SEPARATOR | SWT.HORIZONTAL);
            separator.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

            // Category label
            Label categoryLabel = new Label(toolbarContent, SWT.NONE);
            categoryLabel.setText(category + ":");
            categoryLabel.setFont(boldFont);

            // Buttons for this category
            boolean first = true;
            for (String[] nodeInfo : nodeList) {
                String displayName = nodeInfo[0];
                String registryName = nodeInfo[1];
                createSearchableButton(toolbarContent, displayName, category,
                    () -> addEffectNode(registryName),
                    first ? categoryLabel : null,
                    first ? separator : null,
                    first);
                first = false;
            }
        }

        // Set the toolbar as content of scrolled composite
        scrolledToolbar.setContent(toolbarContent);
        scrolledToolbar.setMinSize(toolbarContent.computeSize(SWT.DEFAULT, SWT.DEFAULT));
    }

    private void createSearchableButton(Composite parent, String nodeName, String category,
            Runnable action, Label categoryLabel, Label separator, boolean isFirstInCategory) {
        // Button text is just the node name (category is already shown in section headers)
        Button btn = new Button(parent, SWT.PUSH | SWT.FLAT);
        btn.setText(nodeName);
        btn.setBackground(new Color(160, 160, 160));
        GridData gd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        gd.heightHint = btn.computeSize(SWT.DEFAULT, SWT.DEFAULT).y + 2;
        // no gd.horizontalIndent = 3;  // Add left/right padding inside button area
        btn.setLayoutData(gd);
        btn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                // Debounce to prevent double-click from creating two nodes
                long now = System.currentTimeMillis();
                if (now - lastNodeButtonClickTime < NODE_BUTTON_DEBOUNCE_MS) return;
                lastNodeButtonClickTime = now;

                action.run();
                // Clear search and show all after adding
                searchBox.setText("");
            }
        });

        // Track for search filtering
        SearchableButton sb = new SearchableButton(btn, nodeName, category, action);
        sb.categoryLabel = categoryLabel;
        sb.separator = separator;
        sb.isFirstInCategory = isFirstInCategory;
        searchableButtons.add(sb);
    }

    private void filterToolbarButtons() {
        String searchText = searchBox.getText().trim().toLowerCase();

        // Track which categories have visible buttons
        java.util.Map<String, Boolean> categoryVisible = new java.util.HashMap<>();

        for (SearchableButton sb : searchableButtons) {
            boolean visible;
            if (searchText.isEmpty()) {
                visible = true;
            } else {
                // Split search text into words (whitespace/punctuation delimited)
                String[] searchWords = searchText.split("[\\s/\\-_.,]+");
                // Create searchable text from node name and category
                String searchableText = (sb.nodeName + " " + sb.category).toLowerCase();

                // All search words must be found as substrings
                visible = true;
                for (String searchWord : searchWords) {
                    if (searchWord.isEmpty()) continue;
                    if (!searchableText.contains(searchWord)) {
                        visible = false;
                        break;
                    }
                }
            }

            // Update button visibility
            sb.button.setVisible(visible);
            ((GridData) sb.button.getLayoutData()).exclude = !visible;

            // Track category visibility
            if (visible) {
                categoryVisible.put(sb.category, true);
            } else if (!categoryVisible.containsKey(sb.category)) {
                categoryVisible.put(sb.category, false);
            }
        }

        // Update category label and separator visibility
        for (SearchableButton sb : searchableButtons) {
            if (sb.isFirstInCategory && sb.categoryLabel != null) {
                boolean catVisible = categoryVisible.getOrDefault(sb.category, false);
                sb.categoryLabel.setVisible(catVisible);
                ((GridData) sb.categoryLabel.getLayoutData()).exclude = !catVisible;
                if (sb.separator != null) {
                    sb.separator.setVisible(catVisible);
                    ((GridData) sb.separator.getLayoutData()).exclude = !catVisible;
                }
            }
        }

        // Re-layout toolbar
        toolbarContent.layout(true);
        scrolledToolbar.setMinSize(toolbarContent.computeSize(SWT.DEFAULT, SWT.DEFAULT));
    }

    private void addSelectedNode() {
        // If we have a valid selection, use it
        if (selectedButtonIndex >= 0 && selectedButtonIndex < searchableButtons.size()) {
            SearchableButton sb = searchableButtons.get(selectedButtonIndex);
            if (sb.button.isVisible()) {
                sb.action.run();
                searchBox.setText("");
                selectedButtonIndex = -1;
                return;
            }
        }
        // Fall back to first visible
        for (SearchableButton sb : searchableButtons) {
            if (sb.button.isVisible()) {
                sb.action.run();
                searchBox.setText("");
                selectedButtonIndex = -1;
                break;
            }
        }
    }

    private void navigateSelection(int direction) {
        // Get list of visible button indices
        java.util.List<Integer> visibleIndices = new java.util.ArrayList<>();
        for (int i = 0; i < searchableButtons.size(); i++) {
            if (searchableButtons.get(i).button.isVisible()) {
                visibleIndices.add(i);
            }
        }

        if (visibleIndices.isEmpty()) return;

        // Find current position in visible list
        int currentPos = -1;
        if (selectedButtonIndex >= 0) {
            currentPos = visibleIndices.indexOf(selectedButtonIndex);
        }

        // Calculate new position
        int newPos;
        if (currentPos == -1) {
            // No current selection - start at first or last depending on direction
            newPos = direction > 0 ? 0 : visibleIndices.size() - 1;
        } else {
            newPos = currentPos + direction;
            // Wrap around
            if (newPos < 0) newPos = visibleIndices.size() - 1;
            if (newPos >= visibleIndices.size()) newPos = 0;
        }

        // Update selection
        int oldIndex = selectedButtonIndex;
        selectedButtonIndex = visibleIndices.get(newPos);

        // Update visual highlighting
        updateButtonHighlighting(oldIndex);

        // Scroll to make selected button visible
        SearchableButton selected = searchableButtons.get(selectedButtonIndex);
        scrolledToolbar.showControl(selected.button);
    }

    private void updateButtonHighlighting(int oldIndex) {
        Color normalColor = new Color(160, 160, 160);
        Color selectedColor = new Color(100, 150, 255);

        // Reset old selection
        if (oldIndex >= 0 && oldIndex < searchableButtons.size()) {
            searchableButtons.get(oldIndex).button.setBackground(normalColor);
        }

        // Highlight new selection
        if (selectedButtonIndex >= 0 && selectedButtonIndex < searchableButtons.size()) {
            searchableButtons.get(selectedButtonIndex).button.setBackground(selectedColor);
        }
    }

    private void clearButtonHighlighting() {
        Color normalColor = new Color(160, 160, 160);
        if (selectedButtonIndex >= 0 && selectedButtonIndex < searchableButtons.size()) {
            searchableButtons.get(selectedButtonIndex).button.setBackground(normalColor);
        }
        selectedButtonIndex = -1;
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
                nodeObj.addProperty("threadPriority", node.getThreadPriority());
                nodeObj.addProperty("workUnitsCompleted", node.getWorkUnitsCompleted());

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
                } else if (node instanceof BlankSourceNode) {
                    nodeObj.addProperty("type", "BlankSource");
                    BlankSourceNode bsn = (BlankSourceNode) node;
                    nodeObj.addProperty("imageWidth", bsn.getImageWidth());
                    nodeObj.addProperty("imageHeight", bsn.getImageHeight());
                    nodeObj.addProperty("colorIndex", bsn.getColorIndex());
                    nodeObj.addProperty("fpsIndex", bsn.getFpsIndex());
                } else if (node instanceof ProcessingNode) {
                    nodeObj.addProperty("type", "Processing");
                    nodeObj.addProperty("name", ((ProcessingNode) node).getName());

                    // Save node-specific properties
                    if (node instanceof GaussianBlurNode) {
                        GaussianBlurNode gbn = (GaussianBlurNode) node;
                        nodeObj.addProperty("kernelSizeX", gbn.getKernelSizeX());
                        nodeObj.addProperty("kernelSizeY", gbn.getKernelSizeY());
                        nodeObj.addProperty("sigmaX", gbn.getSigmaX());
                    } else if (node instanceof BoxBlurNode) {
                        BoxBlurNode bbn = (BoxBlurNode) node;
                        nodeObj.addProperty("kernelSizeX", bbn.getKernelSizeX());
                        nodeObj.addProperty("kernelSizeY", bbn.getKernelSizeY());
                    } else if (node instanceof GrayscaleNode) {
                        GrayscaleNode gn = (GrayscaleNode) node;
                        nodeObj.addProperty("conversionIndex", gn.getConversionIndex());
                    } else if (node instanceof ThresholdNode) {
                        ThresholdNode tn = (ThresholdNode) node;
                        nodeObj.addProperty("threshValue", tn.getThreshValue());
                        nodeObj.addProperty("maxValue", tn.getMaxValue());
                        nodeObj.addProperty("typeIndex", tn.getTypeIndex());
                        nodeObj.addProperty("modifierIndex", tn.getModifierIndex());
                        nodeObj.addProperty("returnedThreshold", tn.getReturnedThreshold());
                    } else if (node instanceof ContoursNode) {
                        ContoursNode cn = (ContoursNode) node;
                        nodeObj.addProperty("thresholdValue", cn.getThresholdValue());
                        nodeObj.addProperty("retrievalMode", cn.getRetrievalMode());
                        nodeObj.addProperty("approxMethod", cn.getApproxMethod());
                        nodeObj.addProperty("thickness", cn.getThickness());
                        nodeObj.addProperty("colorR", cn.getColorR());
                        nodeObj.addProperty("colorG", cn.getColorG());
                        nodeObj.addProperty("colorB", cn.getColorB());
                        nodeObj.addProperty("showOriginal", cn.getShowOriginal());
                        nodeObj.addProperty("sortMethod", cn.getSortMethod());
                        nodeObj.addProperty("minIndex", cn.getMinIndex());
                        nodeObj.addProperty("maxIndex", cn.getMaxIndex());
                        nodeObj.addProperty("drawMode", cn.getDrawMode());
                    } else if (node instanceof BlobDetectorNode) {
                        BlobDetectorNode bdn = (BlobDetectorNode) node;
                        nodeObj.addProperty("minThreshold", bdn.getMinThreshold());
                        nodeObj.addProperty("maxThreshold", bdn.getMaxThreshold());
                        nodeObj.addProperty("showOriginal", bdn.getShowOriginal());
                        nodeObj.addProperty("filterByArea", bdn.isFilterByArea());
                        nodeObj.addProperty("minArea", bdn.getMinArea());
                        nodeObj.addProperty("maxArea", bdn.getMaxArea());
                        nodeObj.addProperty("filterByCircularity", bdn.isFilterByCircularity());
                        nodeObj.addProperty("minCircularity", bdn.getMinCircularity());
                        nodeObj.addProperty("filterByConvexity", bdn.isFilterByConvexity());
                        nodeObj.addProperty("minConvexity", bdn.getMinConvexity());
                        nodeObj.addProperty("filterByInertia", bdn.isFilterByInertia());
                        nodeObj.addProperty("minInertiaRatio", bdn.getMinInertiaRatio());
                        nodeObj.addProperty("filterByColor", bdn.isFilterByColor());
                        nodeObj.addProperty("blobColor", bdn.getBlobColor());
                        nodeObj.addProperty("colorR", bdn.getColorR());
                        nodeObj.addProperty("colorG", bdn.getColorG());
                        nodeObj.addProperty("colorB", bdn.getColorB());
                    } else if (node instanceof HoughCirclesNode) {
                        HoughCirclesNode hcn = (HoughCirclesNode) node;
                        nodeObj.addProperty("showOriginal", hcn.getShowOriginal());
                        nodeObj.addProperty("minDist", hcn.getMinDist());
                        nodeObj.addProperty("param1", hcn.getParam1());
                        nodeObj.addProperty("param2", hcn.getParam2());
                        nodeObj.addProperty("minRadius", hcn.getMinRadius());
                        nodeObj.addProperty("maxRadius", hcn.getMaxRadius());
                        nodeObj.addProperty("thickness", hcn.getThickness());
                        nodeObj.addProperty("drawCenter", hcn.isDrawCenter());
                        nodeObj.addProperty("colorR", hcn.getColorR());
                        nodeObj.addProperty("colorG", hcn.getColorG());
                        nodeObj.addProperty("colorB", hcn.getColorB());
                    } else if (node instanceof HarrisCornersNode) {
                        HarrisCornersNode harn = (HarrisCornersNode) node;
                        nodeObj.addProperty("showOriginal", harn.getShowOriginal());
                        nodeObj.addProperty("drawFeatures", harn.isDrawFeatures());
                        nodeObj.addProperty("blockSize", harn.getBlockSize());
                        nodeObj.addProperty("ksize", harn.getKsize());
                        nodeObj.addProperty("kPercent", harn.getKPercent());
                        nodeObj.addProperty("thresholdPercent", harn.getThresholdPercent());
                        nodeObj.addProperty("markerSize", harn.getMarkerSize());
                        nodeObj.addProperty("colorR", harn.getColorR());
                        nodeObj.addProperty("colorG", harn.getColorG());
                        nodeObj.addProperty("colorB", harn.getColorB());
                    } else if (node instanceof ShiTomasiCornersNode) {
                        ShiTomasiCornersNode stc = (ShiTomasiCornersNode) node;
                        nodeObj.addProperty("maxCorners", stc.getMaxCorners());
                        nodeObj.addProperty("qualityLevel", stc.getQualityLevel());
                        nodeObj.addProperty("minDistance", stc.getMinDistance());
                        nodeObj.addProperty("blockSize", stc.getBlockSize());
                        nodeObj.addProperty("useHarrisDetector", stc.isUseHarrisDetector());
                        nodeObj.addProperty("kPercent", stc.getKPercent());
                        nodeObj.addProperty("markerSize", stc.getMarkerSize());
                        nodeObj.addProperty("colorR", stc.getColorR());
                        nodeObj.addProperty("colorG", stc.getColorG());
                        nodeObj.addProperty("colorB", stc.getColorB());
                        nodeObj.addProperty("drawFeatures", stc.isDrawFeatures());
                    } else if (node instanceof ScharrNode) {
                        ScharrNode sn = (ScharrNode) node;
                        nodeObj.addProperty("directionIndex", sn.getDirectionIndex());
                        nodeObj.addProperty("scalePercent", sn.getScalePercent());
                        nodeObj.addProperty("delta", sn.getDelta());
                    } else if (node instanceof SobelNode) {
                        SobelNode sn = (SobelNode) node;
                        nodeObj.addProperty("dx", sn.getDx());
                        nodeObj.addProperty("dy", sn.getDy());
                        nodeObj.addProperty("kernelSizeIndex", sn.getKernelSizeIndex());
                    } else if (node instanceof LaplacianNode) {
                        LaplacianNode ln = (LaplacianNode) node;
                        nodeObj.addProperty("kernelSizeIndex", ln.getKernelSizeIndex());
                        nodeObj.addProperty("scalePercent", ln.getScalePercent());
                        nodeObj.addProperty("delta", ln.getDelta());
                        nodeObj.addProperty("useAbsolute", ln.isUseAbsolute());
                    } else if (node instanceof GainNode) {
                        GainNode gn = (GainNode) node;
                        nodeObj.addProperty("gain", gn.getGain());
                    } else if (node instanceof Filter2DNode) {
                        Filter2DNode f2d = (Filter2DNode) node;
                        nodeObj.addProperty("kernelSize", f2d.getKernelSize());
                        JsonArray kernelArray = new JsonArray();
                        int[] values = f2d.getKernelValues();
                        if (values != null) {
                            for (int val : values) {
                                kernelArray.add(val);
                            }
                        }
                        nodeObj.add("kernelValues", kernelArray);
                    } else if (node instanceof MorphologyExNode) {
                        MorphologyExNode men = (MorphologyExNode) node;
                        nodeObj.addProperty("operationIndex", men.getOperationIndex());
                        nodeObj.addProperty("shapeIndex", men.getShapeIndex());
                        nodeObj.addProperty("kernelWidth", men.getKernelWidth());
                        nodeObj.addProperty("kernelHeight", men.getKernelHeight());
                        nodeObj.addProperty("iterations", men.getIterations());
                        nodeObj.addProperty("anchorX", men.getAnchorX());
                        nodeObj.addProperty("anchorY", men.getAnchorY());
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
                    } else if (node instanceof RectangleNode) {
                        RectangleNode rn = (RectangleNode) node;
                        nodeObj.addProperty("x1", rn.getX1());
                        nodeObj.addProperty("y1", rn.getY1());
                        nodeObj.addProperty("x2", rn.getX2());
                        nodeObj.addProperty("y2", rn.getY2());
                        nodeObj.addProperty("colorR", rn.getColorR());
                        nodeObj.addProperty("colorG", rn.getColorG());
                        nodeObj.addProperty("colorB", rn.getColorB());
                        nodeObj.addProperty("thickness", rn.getThickness());
                        nodeObj.addProperty("filled", rn.isFilled());
                    } else if (node instanceof CircleNode) {
                        CircleNode cn = (CircleNode) node;
                        nodeObj.addProperty("centerX", cn.getCenterX());
                        nodeObj.addProperty("centerY", cn.getCenterY());
                        nodeObj.addProperty("radius", cn.getRadius());
                        nodeObj.addProperty("colorR", cn.getColorR());
                        nodeObj.addProperty("colorG", cn.getColorG());
                        nodeObj.addProperty("colorB", cn.getColorB());
                        nodeObj.addProperty("thickness", cn.getThickness());
                        nodeObj.addProperty("filled", cn.isFilled());
                    } else if (node instanceof EllipseNode) {
                        EllipseNode en = (EllipseNode) node;
                        nodeObj.addProperty("centerX", en.getCenterX());
                        nodeObj.addProperty("centerY", en.getCenterY());
                        nodeObj.addProperty("axisX", en.getAxisX());
                        nodeObj.addProperty("axisY", en.getAxisY());
                        nodeObj.addProperty("angle", en.getAngle());
                        nodeObj.addProperty("startAngle", en.getStartAngle());
                        nodeObj.addProperty("endAngle", en.getEndAngle());
                        nodeObj.addProperty("colorR", en.getColorR());
                        nodeObj.addProperty("colorG", en.getColorG());
                        nodeObj.addProperty("colorB", en.getColorB());
                        nodeObj.addProperty("thickness", en.getThickness());
                        nodeObj.addProperty("filled", en.isFilled());
                    } else if (node instanceof LineNode) {
                        LineNode ln = (LineNode) node;
                        nodeObj.addProperty("x1", ln.getX1());
                        nodeObj.addProperty("y1", ln.getY1());
                        nodeObj.addProperty("x2", ln.getX2());
                        nodeObj.addProperty("y2", ln.getY2());
                        nodeObj.addProperty("colorR", ln.getColorR());
                        nodeObj.addProperty("colorG", ln.getColorG());
                        nodeObj.addProperty("colorB", ln.getColorB());
                        nodeObj.addProperty("thickness", ln.getThickness());
                    } else if (node instanceof ArrowNode) {
                        ArrowNode an = (ArrowNode) node;
                        nodeObj.addProperty("x1", an.getX1());
                        nodeObj.addProperty("y1", an.getY1());
                        nodeObj.addProperty("x2", an.getX2());
                        nodeObj.addProperty("y2", an.getY2());
                        nodeObj.addProperty("tipLength", an.getTipLength());
                        nodeObj.addProperty("colorR", an.getColorR());
                        nodeObj.addProperty("colorG", an.getColorG());
                        nodeObj.addProperty("colorB", an.getColorB());
                        nodeObj.addProperty("thickness", an.getThickness());
                    } else if (node instanceof CropNode) {
                        CropNode crn = (CropNode) node;
                        nodeObj.addProperty("cropX", crn.getCropX());
                        nodeObj.addProperty("cropY", crn.getCropY());
                        nodeObj.addProperty("cropWidth", crn.getCropWidth());
                        nodeObj.addProperty("cropHeight", crn.getCropHeight());
                    // AddClampNode and SubtractClampNode have no properties to save
                    } else if (node instanceof AddWeightedNode) {
                        AddWeightedNode awn = (AddWeightedNode) node;
                        nodeObj.addProperty("alpha", awn.getAlpha());
                        nodeObj.addProperty("beta", awn.getBeta());
                        nodeObj.addProperty("gamma", awn.getGamma());
                    } else if (node instanceof TextNode) {
                        TextNode tn = (TextNode) node;
                        nodeObj.addProperty("text", tn.getText());
                        nodeObj.addProperty("posX", tn.getPosX());
                        nodeObj.addProperty("posY", tn.getPosY());
                        nodeObj.addProperty("fontIndex", tn.getFontIndex());
                        nodeObj.addProperty("fontScale", tn.getFontScale());
                        nodeObj.addProperty("colorR", tn.getColorR());
                        nodeObj.addProperty("colorG", tn.getColorG());
                        nodeObj.addProperty("colorB", tn.getColorB());
                        nodeObj.addProperty("thickness", tn.getThickness());
                        nodeObj.addProperty("bold", tn.isBold());
                        nodeObj.addProperty("italic", tn.isItalic());
                    } else if (node instanceof HistogramNode) {
                        HistogramNode hn = (HistogramNode) node;
                        nodeObj.addProperty("modeIndex", hn.getModeIndex());
                        nodeObj.addProperty("backgroundMode", hn.getBackgroundMode());
                        nodeObj.addProperty("fillBars", hn.getFillBars());
                        nodeObj.addProperty("lineThickness", hn.getLineThickness());
                        nodeObj.addProperty("queuesInSync", hn.isQueuesInSync());
                    } else if (node instanceof MatchTemplateNode) {
                        MatchTemplateNode mtn = (MatchTemplateNode) node;
                        nodeObj.addProperty("method", mtn.getMethod());
                        nodeObj.addProperty("queuesInSync", mtn.isQueuesInSync());
                        nodeObj.addProperty("outputMode", mtn.getOutputMode());
                        nodeObj.addProperty("rectColorR", mtn.getRectColorR());
                        nodeObj.addProperty("rectColorG", mtn.getRectColorG());
                        nodeObj.addProperty("rectColorB", mtn.getRectColorB());
                        nodeObj.addProperty("rectThickness", mtn.getRectThickness());
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
                connObj.addProperty("inputIndex", conn.inputIndex);
                connObj.addProperty("queueCapacity", conn.getConfiguredCapacity());
                int queueSize = conn.getQueueSize();
                System.out.println("SAVE: Connection " + nodes.indexOf(conn.source) + " -> " +
                                 nodes.indexOf(conn.target) + " queue size: " + queueSize +
                                 " (queue null? " + (conn.getQueue() == null) + ")");
                connObj.addProperty("queueCount", queueSize);
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
                dcObj.addProperty("queueCapacity", dc.getConfiguredCapacity());
                dcObj.addProperty("queueCount", dc.getQueueSize());
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
                rdcObj.addProperty("queueCapacity", rdc.getConfiguredCapacity());
                rdcObj.addProperty("queueCount", rdc.getQueueSize());
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
                fcObj.addProperty("queueCapacity", fc.getConfiguredCapacity());
                fcObj.addProperty("queueCount", fc.getQueueSize());
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
            int nodeIndex = 0;
            for (PipelineNode node : nodes) {
                if (node instanceof FileSourceNode) {
                    ((FileSourceNode) node).saveThumbnailToCache(cacheDir);
                } else if (node instanceof WebcamSourceNode) {
                    ((WebcamSourceNode) node).saveThumbnailToCache(cacheDir, nodeIndex);
                } else if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).saveThumbnailToCache(cacheDir, nodeIndex);
                }
                nodeIndex++;
            }

            currentFilePath = path;
            addToRecentFiles(path);
            shell.setText("OpenCV Pipeline Editor - " + new File(path).getName());
            clearDirty();

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
                    if (node instanceof SourceNode) {
                        ((SourceNode) node).dispose();
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
                        if (nodeObj.has("threadPriority")) {
                            node.setThreadPriority(nodeObj.get("threadPriority").getAsInt());
                        }
                        if (nodeObj.has("workUnitsCompleted")) {
                            node.setWorkUnitsCompleted(nodeObj.get("workUnitsCompleted").getAsLong());
                        }
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
                        if (node != null) {
                            if (nodeObj.has("threadPriority")) {
                                node.setThreadPriority(nodeObj.get("threadPriority").getAsInt());
                            }
                            if (nodeObj.has("workUnitsCompleted")) {
                                node.setWorkUnitsCompleted(nodeObj.get("workUnitsCompleted").getAsLong());
                            }
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
                                if (nodeObj.has("returnedThreshold")) tn.setReturnedThreshold(nodeObj.get("returnedThreshold").getAsDouble());
                            } else if (node instanceof ContoursNode) {
                                ContoursNode cn = (ContoursNode) node;
                                if (nodeObj.has("thresholdValue")) cn.setThresholdValue(nodeObj.get("thresholdValue").getAsInt());
                                if (nodeObj.has("retrievalMode")) cn.setRetrievalMode(nodeObj.get("retrievalMode").getAsInt());
                                if (nodeObj.has("approxMethod")) cn.setApproxMethod(nodeObj.get("approxMethod").getAsInt());
                                if (nodeObj.has("thickness")) cn.setThickness(nodeObj.get("thickness").getAsInt());
                                if (nodeObj.has("colorR")) cn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) cn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) cn.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("showOriginal")) cn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                                if (nodeObj.has("sortMethod")) cn.setSortMethod(nodeObj.get("sortMethod").getAsInt());
                                if (nodeObj.has("minIndex")) cn.setMinIndex(nodeObj.get("minIndex").getAsInt());
                                if (nodeObj.has("maxIndex")) cn.setMaxIndex(nodeObj.get("maxIndex").getAsInt());
                                if (nodeObj.has("drawMode")) cn.setDrawMode(nodeObj.get("drawMode").getAsInt());
                            } else if (node instanceof BlobDetectorNode) {
                                BlobDetectorNode bdn = (BlobDetectorNode) node;
                                if (nodeObj.has("minThreshold")) bdn.setMinThreshold(nodeObj.get("minThreshold").getAsInt());
                                if (nodeObj.has("maxThreshold")) bdn.setMaxThreshold(nodeObj.get("maxThreshold").getAsInt());
                                if (nodeObj.has("showOriginal")) bdn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                                if (nodeObj.has("filterByArea")) bdn.setFilterByArea(nodeObj.get("filterByArea").getAsBoolean());
                                if (nodeObj.has("minArea")) bdn.setMinArea(nodeObj.get("minArea").getAsInt());
                                if (nodeObj.has("maxArea")) bdn.setMaxArea(nodeObj.get("maxArea").getAsInt());
                                if (nodeObj.has("filterByCircularity")) bdn.setFilterByCircularity(nodeObj.get("filterByCircularity").getAsBoolean());
                                if (nodeObj.has("minCircularity")) bdn.setMinCircularity(nodeObj.get("minCircularity").getAsInt());
                                if (nodeObj.has("filterByConvexity")) bdn.setFilterByConvexity(nodeObj.get("filterByConvexity").getAsBoolean());
                                if (nodeObj.has("minConvexity")) bdn.setMinConvexity(nodeObj.get("minConvexity").getAsInt());
                                if (nodeObj.has("filterByInertia")) bdn.setFilterByInertia(nodeObj.get("filterByInertia").getAsBoolean());
                                if (nodeObj.has("minInertiaRatio")) bdn.setMinInertiaRatio(nodeObj.get("minInertiaRatio").getAsInt());
                                if (nodeObj.has("filterByColor")) bdn.setFilterByColor(nodeObj.get("filterByColor").getAsBoolean());
                                if (nodeObj.has("blobColor")) bdn.setBlobColor(nodeObj.get("blobColor").getAsInt());
                                if (nodeObj.has("colorR")) bdn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) bdn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) bdn.setColorB(nodeObj.get("colorB").getAsInt());
                            } else if (node instanceof HoughCirclesNode) {
                                HoughCirclesNode hcn = (HoughCirclesNode) node;
                                if (nodeObj.has("showOriginal")) hcn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                                if (nodeObj.has("minDist")) hcn.setMinDist(nodeObj.get("minDist").getAsInt());
                                if (nodeObj.has("param1")) hcn.setParam1(nodeObj.get("param1").getAsInt());
                                if (nodeObj.has("param2")) hcn.setParam2(nodeObj.get("param2").getAsInt());
                                if (nodeObj.has("minRadius")) hcn.setMinRadius(nodeObj.get("minRadius").getAsInt());
                                if (nodeObj.has("maxRadius")) hcn.setMaxRadius(nodeObj.get("maxRadius").getAsInt());
                                if (nodeObj.has("thickness")) hcn.setThickness(nodeObj.get("thickness").getAsInt());
                                if (nodeObj.has("drawCenter")) hcn.setDrawCenter(nodeObj.get("drawCenter").getAsBoolean());
                                if (nodeObj.has("colorR")) hcn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) hcn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) hcn.setColorB(nodeObj.get("colorB").getAsInt());
                            } else if (node instanceof HarrisCornersNode) {
                                HarrisCornersNode harn = (HarrisCornersNode) node;
                                if (nodeObj.has("showOriginal")) harn.setShowOriginal(nodeObj.get("showOriginal").getAsBoolean());
                                if (nodeObj.has("blockSize")) harn.setBlockSize(nodeObj.get("blockSize").getAsInt());
                                if (nodeObj.has("ksize")) harn.setKsize(nodeObj.get("ksize").getAsInt());
                                if (nodeObj.has("kPercent")) harn.setKPercent(nodeObj.get("kPercent").getAsInt());
                                if (nodeObj.has("thresholdPercent")) harn.setThresholdPercent(nodeObj.get("thresholdPercent").getAsInt());
                                if (nodeObj.has("markerSize")) harn.setMarkerSize(nodeObj.get("markerSize").getAsInt());
                                if (nodeObj.has("colorR")) harn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) harn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) harn.setColorB(nodeObj.get("colorB").getAsInt());
                            } else if (node instanceof ScharrNode) {
                                ScharrNode sn = (ScharrNode) node;
                                if (nodeObj.has("directionIndex")) sn.setDirectionIndex(nodeObj.get("directionIndex").getAsInt());
                                if (nodeObj.has("scalePercent")) sn.setScalePercent(nodeObj.get("scalePercent").getAsInt());
                                if (nodeObj.has("delta")) sn.setDelta(nodeObj.get("delta").getAsInt());
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
                            } else if (node instanceof RectangleNode) {
                                RectangleNode rn = (RectangleNode) node;
                                if (nodeObj.has("x1")) rn.setX1(nodeObj.get("x1").getAsInt());
                                if (nodeObj.has("y1")) rn.setY1(nodeObj.get("y1").getAsInt());
                                if (nodeObj.has("x2")) rn.setX2(nodeObj.get("x2").getAsInt());
                                if (nodeObj.has("y2")) rn.setY2(nodeObj.get("y2").getAsInt());
                                if (nodeObj.has("colorR")) rn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) rn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) rn.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("thickness")) rn.setThickness(nodeObj.get("thickness").getAsInt());
                                if (nodeObj.has("filled")) rn.setFilled(nodeObj.get("filled").getAsBoolean());
                            } else if (node instanceof CircleNode) {
                                CircleNode cn = (CircleNode) node;
                                if (nodeObj.has("centerX")) cn.setCenterX(nodeObj.get("centerX").getAsInt());
                                if (nodeObj.has("centerY")) cn.setCenterY(nodeObj.get("centerY").getAsInt());
                                if (nodeObj.has("radius")) cn.setRadius(nodeObj.get("radius").getAsInt());
                                if (nodeObj.has("colorR")) cn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) cn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) cn.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("thickness")) cn.setThickness(nodeObj.get("thickness").getAsInt());
                                if (nodeObj.has("filled")) cn.setFilled(nodeObj.get("filled").getAsBoolean());
                            } else if (node instanceof EllipseNode) {
                                EllipseNode en = (EllipseNode) node;
                                if (nodeObj.has("centerX")) en.setCenterX(nodeObj.get("centerX").getAsInt());
                                if (nodeObj.has("centerY")) en.setCenterY(nodeObj.get("centerY").getAsInt());
                                if (nodeObj.has("axisX")) en.setAxisX(nodeObj.get("axisX").getAsInt());
                                if (nodeObj.has("axisY")) en.setAxisY(nodeObj.get("axisY").getAsInt());
                                if (nodeObj.has("angle")) en.setAngle(nodeObj.get("angle").getAsInt());
                                if (nodeObj.has("startAngle")) en.setStartAngle(nodeObj.get("startAngle").getAsInt());
                                if (nodeObj.has("endAngle")) en.setEndAngle(nodeObj.get("endAngle").getAsInt());
                                if (nodeObj.has("colorR")) en.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) en.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) en.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("thickness")) en.setThickness(nodeObj.get("thickness").getAsInt());
                                if (nodeObj.has("filled")) en.setFilled(nodeObj.get("filled").getAsBoolean());
                            } else if (node instanceof LineNode) {
                                LineNode ln = (LineNode) node;
                                if (nodeObj.has("x1")) ln.setX1(nodeObj.get("x1").getAsInt());
                                if (nodeObj.has("y1")) ln.setY1(nodeObj.get("y1").getAsInt());
                                if (nodeObj.has("x2")) ln.setX2(nodeObj.get("x2").getAsInt());
                                if (nodeObj.has("y2")) ln.setY2(nodeObj.get("y2").getAsInt());
                                if (nodeObj.has("colorR")) ln.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) ln.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) ln.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("thickness")) ln.setThickness(nodeObj.get("thickness").getAsInt());
                            } else if (node instanceof ArrowNode) {
                                ArrowNode an = (ArrowNode) node;
                                if (nodeObj.has("x1")) an.setX1(nodeObj.get("x1").getAsInt());
                                if (nodeObj.has("y1")) an.setY1(nodeObj.get("y1").getAsInt());
                                if (nodeObj.has("x2")) an.setX2(nodeObj.get("x2").getAsInt());
                                if (nodeObj.has("y2")) an.setY2(nodeObj.get("y2").getAsInt());
                                if (nodeObj.has("tipLength")) an.setTipLength(nodeObj.get("tipLength").getAsDouble());
                                if (nodeObj.has("colorR")) an.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) an.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) an.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("thickness")) an.setThickness(nodeObj.get("thickness").getAsInt());
                            } else if (node instanceof CropNode) {
                                CropNode crn = (CropNode) node;
                                if (nodeObj.has("cropX")) crn.setCropX(nodeObj.get("cropX").getAsInt());
                                if (nodeObj.has("cropY")) crn.setCropY(nodeObj.get("cropY").getAsInt());
                                if (nodeObj.has("cropWidth")) crn.setCropWidth(nodeObj.get("cropWidth").getAsInt());
                                if (nodeObj.has("cropHeight")) crn.setCropHeight(nodeObj.get("cropHeight").getAsInt());
                            // AddClampNode and SubtractClampNode have no properties to load
                            } else if (node instanceof AddWeightedNode) {
                                AddWeightedNode awn = (AddWeightedNode) node;
                                if (nodeObj.has("alpha")) awn.setAlpha(nodeObj.get("alpha").getAsDouble());
                                if (nodeObj.has("beta")) awn.setBeta(nodeObj.get("beta").getAsDouble());
                                if (nodeObj.has("gamma")) awn.setGamma(nodeObj.get("gamma").getAsDouble());
                            } else if (node instanceof TextNode) {
                                TextNode tn = (TextNode) node;
                                if (nodeObj.has("text")) tn.setText(nodeObj.get("text").getAsString());
                                if (nodeObj.has("posX")) tn.setPosX(nodeObj.get("posX").getAsInt());
                                if (nodeObj.has("posY")) tn.setPosY(nodeObj.get("posY").getAsInt());
                                if (nodeObj.has("fontIndex")) tn.setFontIndex(nodeObj.get("fontIndex").getAsInt());
                                if (nodeObj.has("fontScale")) tn.setFontScale(nodeObj.get("fontScale").getAsDouble());
                                if (nodeObj.has("colorR")) tn.setColorR(nodeObj.get("colorR").getAsInt());
                                if (nodeObj.has("colorG")) tn.setColorG(nodeObj.get("colorG").getAsInt());
                                if (nodeObj.has("colorB")) tn.setColorB(nodeObj.get("colorB").getAsInt());
                                if (nodeObj.has("thickness")) tn.setThickness(nodeObj.get("thickness").getAsInt());
                                if (nodeObj.has("italic")) tn.setItalic(nodeObj.get("italic").getAsBoolean());
                            }
                            // InvertNode has no properties to load
                            node.setOnChanged(() -> { markDirty(); executePipeline(); });
                            nodes.add(node);
                        }
                    }
                }

                // Load thumbnails from cache for ProcessingNodes and WebcamSourceNodes
                String cacheDir = getCacheDir(path);
                for (int i = 0; i < nodes.size(); i++) {
                    PipelineNode node = nodes.get(i);
                    if (node instanceof WebcamSourceNode) {
                        ((WebcamSourceNode) node).loadThumbnailFromCache(cacheDir, i);
                    } else if (node instanceof ProcessingNode) {
                        ((ProcessingNode) node).loadThumbnailFromCache(cacheDir, i);
                    }
                }

                // Load connections
                if (root.has("connections")) {
                    JsonArray connsArray = root.getAsJsonArray("connections");
                    for (JsonElement elem : connsArray) {
                        JsonObject connObj = elem.getAsJsonObject();
                        int sourceId = connObj.get("sourceId").getAsInt();
                        int targetId = connObj.get("targetId").getAsInt();
                        int inputIdx = connObj.has("inputIndex") ? connObj.get("inputIndex").getAsInt() : 1;
                        if (sourceId >= 0 && sourceId < nodes.size() &&
                            targetId >= 0 && targetId < nodes.size()) {
                            Connection conn = new Connection(nodes.get(sourceId), nodes.get(targetId), inputIdx);
                            if (connObj.has("queueCapacity")) {
                                conn.setConfiguredCapacity(connObj.get("queueCapacity").getAsInt());
                            }
                            if (connObj.has("queueCount")) {
                                conn.setLastQueueSize(connObj.get("queueCount").getAsInt());
                            }
                            connections.add(conn);
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
                            DanglingConnection dc = new DanglingConnection(nodes.get(sourceId), new Point(freeEndX, freeEndY));
                            if (dcObj.has("queueCapacity")) {
                                dc.setConfiguredCapacity(dcObj.get("queueCapacity").getAsInt());
                            }
                            if (dcObj.has("queueCount")) {
                                dc.setLastQueueSize(dcObj.get("queueCount").getAsInt());
                            }
                            danglingConnections.add(dc);
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
                            ReverseDanglingConnection rdc = new ReverseDanglingConnection(nodes.get(targetId), new Point(freeEndX, freeEndY));
                            if (rdcObj.has("queueCapacity")) {
                                rdc.setConfiguredCapacity(rdcObj.get("queueCapacity").getAsInt());
                            }
                            if (rdcObj.has("queueCount")) {
                                rdc.setLastQueueSize(rdcObj.get("queueCount").getAsInt());
                            }
                            reverseDanglingConnections.add(rdc);
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
                        FreeConnection fc = new FreeConnection(new Point(startEndX, startEndY), new Point(arrowEndX, arrowEndY));
                        if (fcObj.has("queueCapacity")) {
                            fc.setConfiguredCapacity(fcObj.get("queueCapacity").getAsInt());
                        }
                        if (fcObj.has("queueCount")) {
                            fc.setLastQueueSize(fcObj.get("queueCount").getAsInt());
                        }
                        freeConnections.add(fc);
                    }
                }

                currentFilePath = path;
                addToRecentFiles(path);

                canvas.redraw();
                shell.setText("OpenCV Pipeline Editor - " + new File(path).getName());
                updateCanvasSize();

            } catch (Exception e) {
                MessageBox mb = new MessageBox(shell, SWT.ICON_ERROR | SWT.OK);
                mb.setText("Load Error");
                mb.setMessage("Failed to load: " + e.getMessage());
                mb.open();
            }
        }
    }

    private void createNodeButton(Composite parent, String text, Runnable action) {
        Button btn = new Button(parent, SWT.PUSH | SWT.FLAT);
        btn.setText(text);
        btn.setBackground(new Color(160, 160, 160));
        GridData gd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        gd.heightHint = btn.computeSize(SWT.DEFAULT, SWT.DEFAULT).y + 2;
        btn.setLayoutData(gd);
        btn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                action.run();
            }
        });
    }

    /**
     * Update canvas size based on node positions and zoom level.
     * Canvas size = max(viewport size, (max node bounds + padding) * zoom)
     */
    private void updateCanvasSize() {
        if (scrolledCanvas == null || canvas == null) return;

        // Get viewport size
        Rectangle viewportBounds = scrolledCanvas.getClientArea();
        int minWidth = viewportBounds.width > 0 ? viewportBounds.width : 800;
        int minHeight = viewportBounds.height > 0 ? viewportBounds.height : 600;

        // Calculate bounds from all nodes (in canvas coordinates)
        int maxX = 0;
        int maxY = 0;
        for (PipelineNode node : nodes) {
            int nodeRight = node.getX() + node.getWidth();
            int nodeBottom = node.getY() + node.getHeight();
            if (nodeRight > maxX) maxX = nodeRight;
            if (nodeBottom > maxY) maxY = nodeBottom;
        }

        // Add padding and apply zoom
        int padding = 200;
        int requiredWidth = (int) ((maxX + padding) * zoomLevel);
        int requiredHeight = (int) ((maxY + padding) * zoomLevel);

        // Canvas size is max of viewport and required content size
        int canvasWidth = Math.max(minWidth, requiredWidth);
        int canvasHeight = Math.max(minHeight, requiredHeight);

        // Only update if size changed
        Point currentSize = canvas.getSize();
        if (currentSize.x != canvasWidth || currentSize.y != canvasHeight) {
            canvas.setSize(canvasWidth, canvasHeight);
            scrolledCanvas.setMinSize(canvasWidth, canvasHeight);
        }
    }

    private void createCanvas() {
        // Create a composite to hold scrolled canvas and status bar
        Composite canvasContainer = new Composite(sashForm, SWT.NONE);
        GridLayout containerLayout = new GridLayout(1, false);
        containerLayout.marginWidth = 0;
        containerLayout.marginHeight = 0;
        containerLayout.verticalSpacing = 0;
        canvasContainer.setLayout(containerLayout);

        // Create scrolled composite for the canvas
        scrolledCanvas = new ScrolledComposite(canvasContainer, SWT.H_SCROLL | SWT.V_SCROLL | SWT.BORDER);
        scrolledCanvas.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        scrolledCanvas.setExpandHorizontal(true);
        scrolledCanvas.setExpandVertical(true);

        canvas = new Canvas(scrolledCanvas, SWT.DOUBLE_BUFFERED);
        canvas.setBackground(display.getSystemColor(SWT.COLOR_WHITE));

        // Set the canvas as the content of the scrolled composite
        scrolledCanvas.setContent(canvas);

        // Initial canvas size - will be updated dynamically based on content
        updateCanvasSize();

        // Update canvas size when viewport is resized
        scrolledCanvas.addControlListener(new org.eclipse.swt.events.ControlAdapter() {
            @Override
            public void controlResized(org.eclipse.swt.events.ControlEvent e) {
                updateCanvasSize();
            }
        });

        // Status bar at bottom of canvas
        Composite statusComp = new Composite(canvasContainer, SWT.NONE);
        statusComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        statusComp.setLayout(new GridLayout(3, false));
        ((GridLayout)statusComp.getLayout()).marginHeight = 2;
        ((GridLayout)statusComp.getLayout()).marginWidth = 5;
        statusComp.setBackground(new Color(160, 160, 160));

        // Node count on the left
        nodeCountLabel = new Label(statusComp, SWT.NONE);
        nodeCountLabel.setText("Nodes: 0");
        nodeCountLabel.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false));
        nodeCountLabel.setBackground(new Color(160, 160, 160));
        nodeCountLabel.setForeground(display.getSystemColor(SWT.COLOR_BLACK));

        // Pipeline status in the center
        statusBar = new Label(statusComp, SWT.NONE);
        statusBar.setText("Pipeline Stopped");
        statusBar.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, true, false));
        statusBar.setBackground(new Color(160, 160, 160));
        statusBar.setForeground(new Color(180, 0, 0)); // Red for stopped

        // Zoom combo on the right
        zoomCombo = new Combo(statusComp, SWT.DROP_DOWN | SWT.READ_ONLY);
        String[] zoomItems = new String[ZOOM_LEVELS.length];
        int defaultIndex = 3; // 100%
        for (int i = 0; i < ZOOM_LEVELS.length; i++) {
            zoomItems[i] = ZOOM_LEVELS[i] + "%";
            if (ZOOM_LEVELS[i] == 100) defaultIndex = i;
        }
        zoomCombo.setItems(zoomItems);
        zoomCombo.select(defaultIndex);
        // Use system colors for proper light/dark mode support
        zoomCombo.setBackground(display.getSystemColor(SWT.COLOR_LIST_BACKGROUND));
        zoomCombo.setForeground(display.getSystemColor(SWT.COLOR_LIST_FOREGROUND));
        GridData comboGd = new GridData(SWT.RIGHT, SWT.CENTER, false, false);
        comboGd.widthHint = 75;
        zoomCombo.setLayoutData(comboGd);
        zoomCombo.addListener(SWT.Selection, e -> {
            int idx = zoomCombo.getSelectionIndex();
            if (idx >= 0 && idx < ZOOM_LEVELS.length) {
                zoomLevel = ZOOM_LEVELS[idx] / 100.0;
                updateCanvasSize();
                canvas.redraw();
            }
        });

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

        // Ctrl+scroll wheel for zoom
        canvas.addListener(SWT.MouseVerticalWheel, event -> {
            if ((event.stateMask & SWT.MOD1) != 0) { // Ctrl/Cmd held
                int currentIdx = zoomCombo.getSelectionIndex();
                int newIdx;
                if (event.count > 0) {
                    // Zoom in - go to next higher zoom level
                    newIdx = Math.min(ZOOM_LEVELS.length - 1, currentIdx + 1);
                } else {
                    // Zoom out - go to next lower zoom level
                    newIdx = Math.max(0, currentIdx - 1);
                }
                if (newIdx != currentIdx) {
                    zoomCombo.select(newIdx);
                    zoomLevel = ZOOM_LEVELS[newIdx] / 100.0;
                    updateCanvasSize();
                    canvas.redraw();
                }
                event.doit = false; // Consume event
            }
        });

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
                            Point targetPoint = getConnectionTargetPoint(conn);
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
                    markDirty();
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
        // Create vertical SashForm for top panel and preview
        SashForm rightSash = new SashForm(sashForm, SWT.VERTICAL);

        // Top panel - for content moved from left toolbar
        Composite topPanel = new Composite(rightSash, SWT.BORDER);
        topPanel.setLayout(new GridLayout(1, false));

        Label topLabel = new Label(topPanel, SWT.NONE);
        topLabel.setText("Controls");
        Font topBoldFont = new Font(display, "Arial", 11, SWT.BOLD);
        topLabel.setFont(topBoldFont);

        // Start/Stop button for continuous threaded execution
        startStopBtn = new Button(topPanel, SWT.PUSH);
        startStopBtn.setText("Start Pipeline");
        startStopBtn.setBackground(new Color(100, 180, 100)); // Green for start
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

        // Single frame button
        Button runBtn = new Button(topPanel, SWT.PUSH);
        runBtn.setText("Single Frame");
        runBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        runBtn.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                executePipeline();
            }
        });

        // Separator
        new Label(topPanel, SWT.SEPARATOR | SWT.HORIZONTAL)
            .setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Instructions
        Label instructions = new Label(topPanel, SWT.WRAP);
        instructions.setText("Instructions:\n" +
            "• Click node name in left panel to create\n" +
            "• Drag nodes to move\n" +
            "• Click connection circles to connect\n" +
            "• Double-click node for properties\n" +
            "• Single click any node to see its output in the Output Preview");
        GridData instructionsGd = new GridData(SWT.FILL, SWT.FILL, true, true);
        instructionsGd.widthHint = 150;
        instructions.setLayoutData(instructionsGd);

        // Bottom panel - Output Preview
        Composite previewPanel = new Composite(rightSash, SWT.BORDER);
        previewPanel.setLayout(new GridLayout(1, false));

        Label titleLabel = new Label(previewPanel, SWT.NONE);
        titleLabel.setText("Output Preview of Selected Node");
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
                if (selectedNodes.size() > 1) {
                    e.gc.drawString("Select a single node", 10, 10, true);
                    e.gc.drawString("to preview its output", 10, 30, true);
                } else {
                    e.gc.drawString("No output yet", 10, 10, true);
                    e.gc.drawString("Click 'Start Pipeline'", 10, 30, true);
                }
            }
        });

        // Set initial weights (30% top panel, 70% preview)
        rightSash.setWeights(new int[] {30, 70});
    }

    private void executePipeline() {
        // Find all nodes and build execution order
        // For now, simple linear execution following connections

        // Find source node (FileSourceNode, WebcamSourceNode, or BlankSourceNode with no incoming connections)
        PipelineNode sourceNode = null;
        for (PipelineNode node : nodes) {
            if (node instanceof FileSourceNode || node instanceof WebcamSourceNode || node instanceof BlankSourceNode) {
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
        } else if (sourceNode instanceof BlankSourceNode) {
            currentMat = ((BlankSourceNode) sourceNode).getNextFrame();
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

        // Find ALL source nodes (FileSourceNode, WebcamSourceNode, or BlankSourceNode)
        List<PipelineNode> sourceNodes = new ArrayList<>();
        for (PipelineNode node : nodes) {
            if (node instanceof SourceNode) {
                boolean hasIncoming = false;
                for (Connection conn : connections) {
                    if (conn.target == node) {
                        hasIncoming = true;
                        break;
                    }
                }
                if (!hasIncoming) {
                    sourceNodes.add(node);
                }
            }
        }

        // Validate we have at least one source node
        if (sourceNodes.isEmpty()) {
            MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
            mb.setText("No Source");
            mb.setMessage("Please add a source node (Image Source, Webcam, or Blank).");
            mb.open();
            return;
        }

        // Validate FileSourceNodes have loaded images
        for (PipelineNode sourceNode : sourceNodes) {
            if (sourceNode instanceof FileSourceNode fsn) {
                if (fsn.getLoadedImage() == null) {
                    MessageBox mb = new MessageBox(shell, SWT.ICON_WARNING | SWT.OK);
                    mb.setText("No Source");
                    mb.setMessage("Please load an image in the source node first.");
                    mb.open();
                    return;
                }
            }
        }

        // Each source node is a pipeline
        int pipelineCount = sourceNodes.size();

        // If no node is selected, auto-select the terminal node of the first pipeline
        if (selectedNodes.isEmpty()) {
            // Find terminal node of first pipeline
            PipelineNode current = sourceNodes.get(0);
            while (current != null) {
                PipelineNode next = null;
                for (Connection conn : connections) {
                    if (conn.source == current) {
                        next = conn.target;
                        break;
                    }
                }
                if (next == null) {
                    selectedNodes.add(current);
                    break;
                }
                current = next;
            }
            canvas.redraw();
        }

        // Clear old state
        stopPipeline();

        // Activate all connections (creates queues and wires them to nodes)
        for (Connection conn : connections) {
            conn.activate();
        }

        // Activate dangling connections (creates queues and wires to source nodes)
        for (DanglingConnection dc : danglingConnections) {
            dc.activate();
        }

        // Activate reverse dangling connections (creates queues and wires to target nodes)
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            rdc.activate();
        }

        // Activate free connections (creates queues but not wired to any nodes)
        for (FreeConnection fc : freeConnections) {
            fc.activate();
        }

        // Set up input node references for backpressure management
        for (Connection conn : connections) {
            if (conn.inputIndex == 2) {
                // Second input for dual-input nodes
                conn.target.setInputNode2(conn.source);
            } else {
                // Primary input
                conn.target.setInputNode(conn.source);
            }
        }

        // Set up frame callbacks for preview updates
        for (PipelineNode node : nodes) {
            final PipelineNode n = node;
            node.setOnFrameCallback(frame -> {
                if (!display.isDisposed()) {
                    // Clone frame before async call since it will be released
                    Mat frameClone = frame.clone();
                    display.asyncExec(() -> {
                        if (canvas.isDisposed()) {
                            frameClone.release();
                            return;
                        }
                        canvas.redraw();

                        // Update preview if this node is selected
                        if (selectedNodes.size() == 1 && selectedNodes.contains(n)) {
                            updatePreview(frameClone);
                        } else if (selectedNodes.isEmpty() && n.getOutputQueue() == null) {
                            // No selection and this is the last node - show its output
                            updatePreview(frameClone);
                        }

                        // Release the cloned frame after use
                        frameClone.release();
                    });
                }
            });
        }

        pipelineRunning.set(true);
        statusBar.setText("Pipeline Running (" + pipelineCount + " pipeline" + (pipelineCount > 1 ? "s" : "") + ")");
        statusBar.setForeground(new Color(0, 128, 0)); // Green text

        // Start all nodes
        for (PipelineNode node : nodes) {
            node.startProcessing();
        }

        // Update button
        startStopBtn.setText("Stop Pipeline");
        startStopBtn.setBackground(new Color(200, 100, 100)); // Red for stop
    }

    private void stopPipeline() {
        pipelineRunning.set(false);
        statusBar.setText("Pipeline Stopped");
        statusBar.setForeground(new Color(180, 0, 0)); // Red for stopped

        // Stop all nodes
        for (PipelineNode node : nodes) {
            node.stopProcessing();
        }

        // Deactivate all connections (clears queues)
        for (Connection conn : connections) {
            conn.deactivate();
        }

        // Deactivate dangling connections (clears queues)
        for (DanglingConnection dc : danglingConnections) {
            dc.deactivate();
        }

        // Deactivate reverse dangling connections (clears queues)
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            rdc.deactivate();
        }

        // Deactivate free connections (clears queues)
        for (FreeConnection fc : freeConnections) {
            fc.deactivate();
        }

        // Clear frame callbacks and input node references
        for (PipelineNode node : nodes) {
            node.setOnFrameCallback(null);
            node.setInputNode(null);
            node.setInputNode2(null);
        }

        // Update button
        if (startStopBtn != null && !startStopBtn.isDisposed()) {
            startStopBtn.setText("Start Pipeline");
            startStopBtn.setBackground(new Color(100, 180, 100)); // Green for start
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
        if (mat == null || mat.empty()) {
            return;
        }

        // Dispose old preview image
        if (previewImage != null && !previewImage.isDisposed()) {
            previewImage.dispose();
        }

        // Ensure Mat is 8-bit (normalize if floating-point)
        Mat mat8u = new Mat();
        int depth = mat.depth();
        if (depth != org.opencv.core.CvType.CV_8U && depth != org.opencv.core.CvType.CV_8S) {
            // Floating-point or other type - normalize to 8-bit
            Core.normalize(mat, mat8u, 0, 255, Core.NORM_MINMAX, org.opencv.core.CvType.CV_8U);
        } else {
            mat8u = mat;
        }

        // Convert Mat to SWT Image
        Mat rgb = new Mat();
        if (mat8u.channels() == 3) {
            Imgproc.cvtColor(mat8u, rgb, Imgproc.COLOR_BGR2RGB);
        } else if (mat8u.channels() == 1) {
            Imgproc.cvtColor(mat8u, rgb, Imgproc.COLOR_GRAY2RGB);
        } else {
            rgb = mat8u;
        }

        int width = rgb.width();
        int height = rgb.height();
        byte[] data = new byte[width * height * 3];
        rgb.get(0, 0, data);

        // Create ImageData with direct data copy (accounts for scanline padding)
        PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
        ImageData imageData = new ImageData(width, height, 24, palette);

        // Copy data row by row to handle scanline padding
        int bytesPerLine = imageData.bytesPerLine;
        for (int row = 0; row < height; row++) {
            int srcOffset = row * width * 3;
            int dstOffset = row * bytesPerLine;
            for (int col = 0; col < width; col++) {
                int srcIdx = srcOffset + col * 3;
                int dstIdx = dstOffset + col * 3;
                // Direct copy - data is already RGB from cvtColor
                imageData.data[dstIdx] = data[srcIdx];         // R
                imageData.data[dstIdx + 1] = data[srcIdx + 1]; // G
                imageData.data[dstIdx + 2] = data[srcIdx + 2]; // B
            }
        }

        previewImage = new Image(display, imageData);
        previewCanvas.redraw();

        // Clean up temporary mat8u if it was created
        if (mat8u != mat && !mat8u.empty()) {
            mat8u.release();
        }
    }

    private void updatePreviewFromSelection() {
        // If exactly one node is selected, show its output
        if (selectedNodes.size() == 1) {
            PipelineNode selected = selectedNodes.iterator().next();
            Mat outputMat = selected.getOutputMat();
            if (outputMat != null && !outputMat.empty()) {
                updatePreview(outputMat);
            }
        } else if (selectedNodes.size() > 1) {
            // Multiple nodes selected - clear preview
            if (previewImage != null && !previewImage.isDisposed()) {
                previewImage.dispose();
                previewImage = null;
            }
            previewCanvas.redraw();
        }
    }

    private void updateNodeCount() {
        if (nodeCountLabel != null && !nodeCountLabel.isDisposed()) {
            nodeCountLabel.setText("Nodes: " + nodes.size());
            nodeCountLabel.getParent().layout();
        }
    }

    private void paintCanvas(GC gc) {
        gc.setAntialias(SWT.ON);
        updateNodeCount();

        // Draw grid background (not scaled - stays fixed)
        Rectangle bounds = canvas.getClientArea();
        int gridSize = 20;
        gc.setForeground(new Color(230, 230, 230));
        gc.setLineWidth(1);
        for (int x = 0; x < bounds.width; x += gridSize) {
            gc.drawLine(x, 0, x, bounds.height);
        }
        for (int y = 0; y < bounds.height; y += gridSize) {
            gc.drawLine(0, y, bounds.width, y);
        }

        // Apply zoom transform for all content
        Transform transform = new Transform(display);
        transform.scale((float)zoomLevel, (float)zoomLevel);
        gc.setTransform(transform);

        // Draw connections first (so nodes appear on top)
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint();
            Point end = getConnectionTargetPoint(conn);

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

            // Draw queue size (always show, whether running or not)
            int queueSize = conn.getQueueSize();
            int midX = (start.x + end.x) / 2;
            int midY = (start.y + end.y) / 2;
            String sizeText = String.valueOf(queueSize);
            gc.setForeground(display.getSystemColor(SWT.COLOR_WHITE));
            gc.setBackground(display.getSystemColor(SWT.COLOR_DARK_BLUE));
            Point textExtent = gc.textExtent(sizeText);
            gc.fillRoundRectangle(midX - textExtent.x/2 - 3, midY - textExtent.y/2 - 2,
                textExtent.x + 6, textExtent.y + 4, 6, 6);
            gc.drawString(sizeText, midX - textExtent.x/2, midY - textExtent.y/2, true);
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

        // Draw nodes on top of connections
        // Draw selection highlight underneath each node, then the node itself
        for (PipelineNode node : nodes) {
            node.drawSelectionHighlight(gc, selectedNodes.contains(node));
            node.paint(gc);
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

        // Clean up transform
        gc.setTransform(null);
        transform.dispose();
    }

    // Helper to get the correct input point for a connection based on its inputIndex
    private Point getConnectionTargetPoint(Connection conn) {
        if (conn.inputIndex == 2 && conn.target instanceof DualInputNode) {
            return ((DualInputNode) conn.target).getInputPoint2();
        }
        return conn.target.getInputPoint();
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
            // Convert to canvas coordinates accounting for zoom
            Point clickPoint = toCanvasPoint(e.x, e.y);
            int radius = 8; // Slightly larger than visual for easier clicking
            boolean cmdHeld = (e.stateMask & SWT.MOD1) != 0;

            // First check if clicking on an input connection point (to yank off existing connection)
            // Iterate in reverse for z-order - last added is on top
            for (int i = nodes.size() - 1; i >= 0; i--) {
                PipelineNode node = nodes.get(i);

                // Check for second input point on dual-input nodes first
                if (node instanceof DualInputNode) {
                    Point inputPoint2 = ((DualInputNode) node).getInputPoint2();
                    double dist2 = Math.sqrt(Math.pow(clickPoint.x - inputPoint2.x, 2) +
                                           Math.pow(clickPoint.y - inputPoint2.y, 2));
                    if (dist2 <= radius) {
                        // Check if this connection point is obscured by another node on top
                        boolean obscured = false;
                        for (int j = i + 1; j < nodes.size(); j++) {
                            if (nodes.get(j).containsPoint(clickPoint)) {
                                obscured = true;
                                break;
                            }
                        }
                        if (obscured) continue;
                        // Check if there's a connection to this second input
                        Connection connToRemove = null;
                        for (Connection conn : connections) {
                            if (conn.target == node && conn.inputIndex == 2) {
                                connToRemove = conn;
                                break;
                            }
                        }
                        if (connToRemove != null) {
                            // Yank off the connection - remove it and start dragging from the source
                            connectionSource = connToRemove.source;
                            connectionEndPoint = clickPoint;
                            targetInputIndex = 2;
                            connections.remove(connToRemove);
                            canvas.redraw();
                            return;
                        }
                        // No existing connection - start a new connection from second input point (reverse direction)
                        connectionTarget = node;
                        targetInputIndex = 2;
                        connectionEndPoint = clickPoint;
                        canvas.redraw();
                        return;
                    }
                }

                Point inputPoint = node.getInputPoint();
                double dist = Math.sqrt(Math.pow(clickPoint.x - inputPoint.x, 2) +
                                       Math.pow(clickPoint.y - inputPoint.y, 2));
                if (dist <= radius) {
                    // Check if this connection point is obscured by another node on top
                    boolean obscured = false;
                    for (int j = i + 1; j < nodes.size(); j++) {
                        if (nodes.get(j).containsPoint(clickPoint)) {
                            obscured = true;
                            break;
                        }
                    }
                    if (obscured) continue;
                    // Check if there's a connection to this input
                    Connection connToRemove = null;
                    for (Connection conn : connections) {
                        if (conn.target == node && conn.inputIndex == 1) {
                            connToRemove = conn;
                            break;
                        }
                    }
                    if (connToRemove != null) {
                        // Yank off the connection - remove it and start dragging from the source
                        connectionSource = connToRemove.source;
                        connectionEndPoint = clickPoint;
                        targetInputIndex = 1;
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
                    targetInputIndex = 1;
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
            // Iterate in reverse for z-order - last added is on top
            for (int i = nodes.size() - 1; i >= 0; i--) {
                PipelineNode node = nodes.get(i);
                Point outputPoint = node.getOutputPoint();
                double dist = Math.sqrt(Math.pow(clickPoint.x - outputPoint.x, 2) +
                                       Math.pow(clickPoint.y - outputPoint.y, 2));
                if (dist <= radius) {
                    // Check if this connection point is obscured by another node on top
                    boolean obscured = false;
                    for (int j = i + 1; j < nodes.size(); j++) {
                        if (nodes.get(j).containsPoint(clickPoint)) {
                            obscured = true;
                            break;
                        }
                    }
                    if (obscured) continue;
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

            // Check for node selection and dragging (iterate in reverse for z-order - last added is on top)
            for (int i = nodes.size() - 1; i >= 0; i--) {
                PipelineNode node = nodes.get(i);
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
                    dragOffset = new Point(clickPoint.x - node.x, clickPoint.y - node.y);
                    isDragging = true;

                    // Update preview to show selected node's output
                    updatePreviewFromSelection();

                    canvas.redraw();
                    return;
                }
            }

            // Check for connection line selection (clicking on the line itself, not endpoints)
            double clickThreshold = 5.0; // Distance threshold for click detection

            // Check regular connections
            for (Connection conn : connections) {
                Point start = conn.source.getOutputPoint();
                Point end = getConnectionTargetPoint(conn);
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
        // Convert to canvas coordinates accounting for zoom
        Point clickPoint = toCanvasPoint(e.x, e.y);

        if (connectionSource != null) {
            boolean connected = false;
            PipelineNode targetNode = null;
            int inputIdx = 1;

            // Check if dropped on an input point (check second input first for dual-input nodes)
            for (PipelineNode node : nodes) {
                if (node != connectionSource) {
                    // Check second input point for dual-input nodes first
                    if (node instanceof DualInputNode) {
                        Point inputPoint2 = ((DualInputNode) node).getInputPoint2();
                        int radius = 8;
                        double dist2 = Math.sqrt(Math.pow(clickPoint.x - inputPoint2.x, 2) +
                                               Math.pow(clickPoint.y - inputPoint2.y, 2));
                        if (dist2 <= radius) {
                            targetNode = node;
                            inputIdx = 2;
                            connected = true;
                            break;
                        }
                    }

                    Point inputPoint = node.getInputPoint();
                    int radius = 8;
                    double dist = Math.sqrt(Math.pow(clickPoint.x - inputPoint.x, 2) +
                                           Math.pow(clickPoint.y - inputPoint.y, 2));
                    if (dist <= radius) {
                        targetNode = node;
                        inputIdx = 1;
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
                        inputIdx = 1;
                        connected = true;
                        break;
                    }
                }
            }

            if (connected && targetNode != null) {
                // Create a new connection
                connections.add(new Connection(connectionSource, targetNode, inputIdx));
                markDirty();
            } else if (connectionEndPoint != null) {
                // Create a dangling connection
                danglingConnections.add(new DanglingConnection(connectionSource, connectionEndPoint));
                markDirty();
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
                // Create a new connection with the correct input index
                connections.add(new Connection(sourceNode, connectionTarget, targetInputIndex));
                markDirty();
            } else if (connectionEndPoint != null) {
                // Create a reverse dangling connection
                reverseDanglingConnections.add(new ReverseDanglingConnection(connectionTarget, connectionEndPoint));
                markDirty();
            }

            connectionTarget = null;
            connectionEndPoint = null;
            targetInputIndex = 1; // Reset to default
            canvas.redraw();

            if (connected) {
                executePipeline();
            }
        }

        // Handle free connection dragging (both ends unattached)
        if (freeConnectionFixedEnd != null) {
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
                Point end = getConnectionTargetPoint(conn);
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

        // Mark dirty if nodes were actually moved
        if (nodesMoved) {
            markDirty();
            updateCanvasSize();
            nodesMoved = false;
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

        // Convert to canvas coordinates accounting for zoom
        int canvasX = toCanvasX(e.x);
        int canvasY = toCanvasY(e.y);

        if (isDragging && selectedNode != null) {
            // Calculate the delta movement
            int deltaX = canvasX - dragOffset.x - selectedNode.x;
            int deltaY = canvasY - dragOffset.y - selectedNode.y;

            // Only mark as moved if there's actual movement
            if (deltaX != 0 || deltaY != 0) {
                nodesMoved = true;
            }

            // Move all selected nodes by the same delta
            if (selectedNodes.contains(selectedNode) && selectedNodes.size() > 1) {
                for (PipelineNode node : selectedNodes) {
                    node.x += deltaX;
                    node.y += deltaY;
                }
            } else {
                // Single node drag
                selectedNode.x = canvasX - dragOffset.x;
                selectedNode.y = canvasY - dragOffset.y;
            }
            canvas.redraw();
        } else if (connectionSource != null || connectionTarget != null || freeConnectionFixedEnd != null) {
            connectionEndPoint = new Point(canvasX, canvasY);
            canvas.redraw();
        } else if (isSelectionBoxDragging) {
            selectionBoxEnd = new Point(canvasX, canvasY);
            canvas.redraw();
        }
    }

    private void handleDoubleClick(MouseEvent e) {
        // Convert to canvas coordinates accounting for zoom
        Point clickPoint = toCanvasPoint(e.x, e.y);
        for (PipelineNode node : nodes) {
            if (node.containsPoint(clickPoint)) {
                if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).showPropertiesDialog();
                } else if (node instanceof SourceNode) {
                    ((SourceNode) node).showPropertiesDialog();
                }
                return;
            }
        }
    }

    private void handleRightClick(MenuDetectEvent e) {
        // Convert to canvas coordinates accounting for zoom
        Point screenPoint = display.map(null, canvas, new Point(e.x, e.y));
        Point clickPoint = toCanvasPoint(screenPoint.x, screenPoint.y);

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
                } else if (node instanceof BlankSourceNode) {
                    MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
                    editItem.setText("Edit Properties...");
                    editItem.addListener(SWT.Selection, evt -> {
                        ((BlankSourceNode) node).showPropertiesDialog();
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
            Point end = getConnectionTargetPoint(conn);
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
        markDirty();
        updateCanvasSize();
        canvas.redraw();
    }

    private void addWebcamSourceNode() {
        addWebcamSourceNodeAt(50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addWebcamSourceNodeAt(int x, int y) {
        WebcamSourceNode node = new WebcamSourceNode(shell, display, canvas, x, y);
        nodes.add(node);
        markDirty();
        updateCanvasSize();
        canvas.redraw();
    }

    private void addBlankSourceNode() {
        addBlankSourceNodeAt(50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addBlankSourceNodeAt(int x, int y) {
        BlankSourceNode node = new BlankSourceNode(shell, display, x, y);
        nodes.add(node);
        markDirty();
        updateCanvasSize();
        canvas.redraw();
    }

    private void addEffectNode(String type) {
        addEffectNodeAt(type, 50 + nodes.size() * 30, 50 + nodes.size() * 30);
    }

    private void addEffectNodeAt(String type, int x, int y) {
        ProcessingNode node = createEffectNode(type, x, y);
        if (node != null) {
            node.setOnChanged(() -> { markDirty(); executePipeline(); });
            nodes.add(node);
            markDirty();
            updateCanvasSize();
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
            case "CLAHE":
                return "CLAHE Contrast";
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
            case "Gradient/MorphX":
                return "MorphologyEx";

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
            case "Crop":
                return "Crop";

            // Dual Input Nodes
            case "Add w/Clamp":
                return "AddClamp";
            case "Subtract w/Clamp":
            case "Sub w/Clamp":  // Old name for backward compatibility
                return "SubtractClamp";
            case "Bitwise AND":
                return "BitwiseAnd";
            case "Bitwise OR":
                return "BitwiseOr";
            case "Bitwise XOR":
                return "BitwiseXor";

            // Filter
            case "Bitwise NOT":
                return "BitwiseNot";
            case "Filter2D w/Kernel":
                return "Filter2D";
            case "FFT High-Pass Filter":
                return "FFTHighPass";
            case "FFT Low-Pass Filter":
                return "FFTLowPass";
            case "Bit Planes Grayscale":
                return "BitPlanesGrayscale";
            case "Bit Planes Color":
                return "BitPlanesColor";

            default:
                return type;
        }
    }
}
