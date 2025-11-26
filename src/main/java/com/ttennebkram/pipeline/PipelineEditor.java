package com.ttennebkram.pipeline;

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
import com.ttennebkram.pipeline.nodes.*;
import com.ttennebkram.pipeline.model.*;
import com.ttennebkram.pipeline.registry.NodeRegistry;
import com.ttennebkram.pipeline.serialization.PipelineSerializer;

public class PipelineEditor {

    // Static initialization - auto-discover all node types
    static {
        NodeRegistry.initialize();
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
    private int connectionSourceOutputIndex = 0; // Which output to connect from (for multi-output nodes)
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

            // Load via PipelineSerializer
            PipelineSerializer.PipelineDocument doc = PipelineSerializer.load(path, display, shell, canvas);

            // Copy loaded data to this editor's collections
            nodes.addAll(doc.nodes);
            connections.addAll(doc.connections);
            danglingConnections.addAll(doc.danglingConnections);
            reverseDanglingConnections.addAll(doc.reverseDanglingConnections);
            freeConnections.addAll(doc.freeConnections);

            // Set up change callbacks for ProcessingNodes
            for (PipelineNode node : nodes) {
                if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).setOnChanged(() -> { markDirty(); executePipeline(); });
                }
            }

            // Load thumbnails from cache
            String cacheDir = getCacheDir(path);
            for (int i = 0; i < nodes.size(); i++) {
                nodes.get(i).loadThumbnailFromCache(cacheDir, i);
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

        for (com.ttennebkram.pipeline.registry.NodeRegistry.NodeRegistration info :
             com.ttennebkram.pipeline.registry.NodeRegistry.getAllNodes()) {
            // Skip source nodes - they have their own toolbar section
            if (SourceNode.class.isAssignableFrom(info.nodeClass)) {
                continue;
            }
            // Create temp node to get display name and category
            ProcessingNode tempNode = com.ttennebkram.pipeline.registry.NodeRegistry.createProcessingNode(
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
                // Don't clear search - user may want to add more of the same node
                // Press Esc to clear search
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
                // Don't clear search - user may want to add more of the same node
                // Press Esc to clear search
                return;
            }
        }
        // Fall back to first visible
        for (SearchableButton sb : searchableButtons) {
            if (sb.button.isVisible()) {
                sb.action.run();
                // Don't clear search - user may want to add more of the same node
                // Press Esc to clear search
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
            // Save via PipelineSerializer
            PipelineSerializer.save(path, nodes, connections, danglingConnections,
                                    reverseDanglingConnections, freeConnections);

            // Save thumbnails to cache directory
            String cacheDir = getCacheDir(path);
            for (int i = 0; i < nodes.size(); i++) {
                nodes.get(i).saveThumbnailToCache(cacheDir, i);
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
            loadDiagramFromPath(path);
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

            // Global Esc to clear search (works from anywhere when main shell is active)
            if (event.keyCode == SWT.ESC) {
                // Only clear search if a dialog is not open (active shell is our main shell)
                Shell activeShell = display.getActiveShell();
                if (activeShell == shell && searchBox != null && !searchBox.isDisposed()
                        && !searchBox.getText().isEmpty()) {
                    searchBox.setText("");
                    event.doit = false;
                    return;
                }
            }

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

        // Find source node (any SourceNode with no incoming connections)
        SourceNode sourceNode = null;
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
                    sourceNode = (SourceNode) node;
                    break;
                }
            }
        }

        if (sourceNode == null) {
            // No source node - silently return (may be loading empty diagram)
            return;
        }

        // Get current frame from source
        Mat currentMat = sourceNode.getNextFrame();

        if (currentMat == null || currentMat.empty()) {
            return;
        }

        // Clone the mat so we don't modify the original
        currentMat = currentMat.clone();

        // Set output on source node
        sourceNode.setOutputMat(currentMat);

        // Follow connections and execute each node (handles branching for multi-output)
        executeNodeChain(sourceNode, currentMat, new java.util.HashSet<>());

        // Update preview and redraw canvas to show thumbnails
        canvas.redraw();
    }

    /**
     * Recursively execute a chain of nodes from the given node.
     * Handles multi-output nodes by following all output connections.
     */
    private void executeNodeChain(PipelineNode node, Mat inputMat, java.util.Set<PipelineNode> visited) {
        if (node == null || inputMat == null || visited.contains(node)) {
            return;
        }
        visited.add(node);

        // Find all connections from this node (may have multiple for multi-output nodes)
        java.util.List<Connection> outConnections = new java.util.ArrayList<>();
        for (Connection conn : connections) {
            if (conn.source == node) {
                outConnections.add(conn);
            }
        }

        if (outConnections.isEmpty()) {
            // This is a terminal node - update preview
            updatePreview(inputMat);
            return;
        }

        // Process each output connection
        for (Connection conn : outConnections) {
            PipelineNode nextNode = conn.target;
            if (nextNode == null || visited.contains(nextNode)) continue;

            Mat outputMat = inputMat;

            // Execute the processing
            if (nextNode instanceof ProcessingNode) {
                ProcessingNode procNode = (ProcessingNode) nextNode;
                // Clone input so each branch gets its own copy
                outputMat = procNode.process(inputMat.clone());
                // Set output on this node for thumbnail
                nextNode.setOutputMat(outputMat != null ? outputMat.clone() : null);
            }

            // Continue down this branch
            if (outputMat != null) {
                executeNodeChain(nextNode, outputMat, visited);
            }
        }
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

        // Reset all node counters
        for (PipelineNode node : nodes) {
            node.setWorkUnitsCompleted(0);
            node.setInputReads1(0);
            node.setInputReads2(0);
        }

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
                // Frame is a clone owned by this callback - we must release it when done
                if (frame == null || frame.empty()) {
                    if (frame != null) frame.release();
                    return;
                }
                if (!display.isDisposed()) {
                    // Clone for the async UI thread - must be done synchronously before frame is released
                    Mat uiClone;
                    try {
                        uiClone = frame.clone();
                    } catch (Exception e) {
                        // Clone failed - frame may have been released by another thread
                        return;
                    } finally {
                        // Always release the frame passed to this callback
                        frame.release();
                    }

                    display.asyncExec(() -> {
                        try {
                            if (canvas.isDisposed() || uiClone.empty()) {
                                return;
                            }
                            canvas.redraw();

                            // Update preview if this node is selected
                            if (selectedNodes.size() == 1 && selectedNodes.contains(n)) {
                                updatePreview(uiClone);
                            } else if (selectedNodes.isEmpty() && n.getOutputQueue() == null) {
                                // No selection and this is the last node - show its output
                                updatePreview(uiClone);
                            }
                        } finally {
                            // Always release the UI clone
                            uiClone.release();
                        }
                    });
                } else {
                    // Display is disposed - just release the frame
                    frame.release();
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

        // Apply zoom transform for all content (including grid)
        Transform transform = new Transform(display);
        transform.scale((float)zoomLevel, (float)zoomLevel);
        gc.setTransform(transform);

        // Draw grid background (scales with zoom)
        Rectangle bounds = canvas.getClientArea();
        int gridSize = 20;
        // Calculate canvas area in world coordinates (accounting for zoom)
        int worldWidth = (int)(bounds.width / zoomLevel) + gridSize;
        int worldHeight = (int)(bounds.height / zoomLevel) + gridSize;
        gc.setForeground(new Color(230, 230, 230));
        gc.setLineWidth(1);
        for (int x = 0; x < worldWidth; x += gridSize) {
            gc.drawLine(x, 0, x, worldHeight);
        }
        for (int y = 0; y < worldHeight; y += gridSize) {
            gc.drawLine(0, y, worldWidth, y);
        }

        // Draw connections first (so nodes appear on top)
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint(conn.outputIndex);
            Point end = getConnectionTargetPoint(conn);

            // Highlight selected connections
            if (selectedConnections.contains(conn)) {
                gc.setLineWidth(3);
                gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
            } else {
                gc.setLineWidth(2);
                gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
            }

            // Calculate routed path around obstacles
            java.util.List<Point> path = calculateRoutedPath(start, end, conn.source, conn.target);
            drawRoutedPath(gc, path, conn.source, conn.target);

            // Draw arrow - for bezier, arrow comes from control point direction
            int[] cp = calculateBezierControlPoints(start, end, conn.source, conn.target);
            Point arrowFrom = new Point(cp[2], cp[3]); // Use second control point for arrow direction
            drawArrow(gc, arrowFrom, end);

            // Draw queue size and total frames (always show, whether running or not)
            int queueSize = conn.getQueueSize();
            long totalFrames = conn.getTotalFramesSent();
            Point midPoint = getPathMidpoint(path, conn.source, conn.target);
            String sizeText = String.format("%,d / %,d", queueSize, totalFrames);
            gc.setForeground(display.getSystemColor(SWT.COLOR_WHITE));
            gc.setBackground(display.getSystemColor(SWT.COLOR_DARK_BLUE));
            Point textExtent = gc.textExtent(sizeText);
            gc.fillRoundRectangle(midPoint.x - textExtent.x/2 - 3, midPoint.y - textExtent.y/2 - 2,
                textExtent.x + 6, textExtent.y + 4, 6, 6);
            gc.drawString(sizeText, midPoint.x - textExtent.x/2, midPoint.y - textExtent.y/2, true);
        }

        // Draw dangling connections with dashed lines
        gc.setLineStyle(SWT.LINE_DASH);
        for (DanglingConnection dangling : danglingConnections) {
            Point start = dangling.source.getOutputPoint(dangling.outputIndex);
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
            Point start = connectionSource.getOutputPoint(connectionSourceOutputIndex);
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
        if (conn.inputIndex == 2 && conn.target.hasDualInput()) {
            return conn.target.getInputPoint2();
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

    /**
     * Calculate a routed path from start to end that avoids nodes.
     * The path connects at connection points on node edges but doesn't pass through node bodies.
     */
    private java.util.List<Point> calculateRoutedPath(Point start, Point end, PipelineNode sourceNode, PipelineNode targetNode) {
        java.util.List<Point> path = new java.util.ArrayList<>();
        path.add(start);

        int margin = 15; // Margin around nodes for routing

        // Calculate exit point (just past the source node's right edge)
        int exitX = sourceNode.x + sourceNode.width + margin;
        // Calculate entry point (just before the target node's left edge)
        int entryX = targetNode.x - margin;

        // Collect ALL nodes as potential obstacles (including source and target)
        // for checking the vertical segment
        java.util.List<PipelineNode> allObstacles = new java.util.ArrayList<>(nodes);

        // Find the bounds of all nodes that overlap with the horizontal span
        int topMost = Integer.MAX_VALUE;
        int bottomMost = Integer.MIN_VALUE;

        for (PipelineNode node : allObstacles) {
            int nodeLeft = node.x - margin;
            int nodeRight = node.x + node.width + margin;

            // Check if node overlaps horizontally with the routing area
            if (nodeRight >= exitX && nodeLeft <= entryX) {
                topMost = Math.min(topMost, node.y - margin);
                bottomMost = Math.max(bottomMost, node.y + node.height + margin);
            }
        }

        // Check if any obstacle blocks a direct vertical segment at exitX or entryX
        boolean needsDetour = false;
        int minPathY = Math.min(start.y, end.y);
        int maxPathY = Math.max(start.y, end.y);

        for (PipelineNode node : allObstacles) {
            // Skip source and target for vertical segment check
            // (we'll route around them with the exit/entry points)
            if (node == sourceNode || node == targetNode) {
                continue;
            }

            int nodeLeft = node.x - margin;
            int nodeRight = node.x + node.width + margin;
            int nodeTop = node.y - margin;
            int nodeBottom = node.y + node.height + margin;

            // Check if a vertical segment at exitX would pass through this node
            if (exitX >= nodeLeft && exitX <= nodeRight) {
                if (!(maxPathY < nodeTop || minPathY > nodeBottom)) {
                    needsDetour = true;
                    break;
                }
            }

            // Check if a vertical segment at entryX would pass through this node
            if (entryX >= nodeLeft && entryX <= nodeRight) {
                if (!(maxPathY < nodeTop || minPathY > nodeBottom)) {
                    needsDetour = true;
                    break;
                }
            }

            // Check if the horizontal segment at start.y or end.y would pass through
            if (start.y >= nodeTop && start.y <= nodeBottom) {
                if (exitX < nodeRight && entryX > nodeLeft) {
                    needsDetour = true;
                    break;
                }
            }
            if (end.y >= nodeTop && end.y <= nodeBottom) {
                if (exitX < nodeRight && entryX > nodeLeft) {
                    needsDetour = true;
                    break;
                }
            }
        }

        if (!needsDetour && start.y == end.y) {
            // Same Y level and no obstacles - simple straight line via exit/entry points
            path.add(new Point(exitX, start.y));
            path.add(new Point(entryX, end.y));
        } else {
            // Need to find a clear Y level for the horizontal segment
            // This Y must be outside BOTH source and target node bodies

            // Calculate the vertical bounds we need to avoid (source and target nodes)
            int sourceTop = sourceNode.y - margin;
            int sourceBottom = sourceNode.y + sourceNode.height + margin;
            int targetTop = targetNode.y - margin;
            int targetBottom = targetNode.y + targetNode.height + margin;

            // Find a routeY that's clear of both source and target
            int routeY;

            // Option 1: Go above both nodes
            int aboveY = Math.min(sourceTop, targetTop) - 10;
            // Option 2: Go below both nodes
            int belowY = Math.max(sourceBottom, targetBottom) + 10;
            // Option 3: Go between them (if there's a gap)
            int betweenY = -1;
            if (sourceBottom < targetTop - 20) {
                // Gap between source (above) and target (below)
                betweenY = (sourceBottom + targetTop) / 2;
            } else if (targetBottom < sourceTop - 20) {
                // Gap between target (above) and source (below)
                betweenY = (targetBottom + sourceTop) / 2;
            }

            // Also consider middle obstacles for the route
            if (needsDetour && topMost != Integer.MAX_VALUE) {
                aboveY = Math.min(aboveY, topMost - 10);
            }
            if (needsDetour && bottomMost != Integer.MIN_VALUE) {
                belowY = Math.max(belowY, bottomMost + 10);
            }

            // Choose the shortest path
            int distAbove = Math.abs(start.y - aboveY) + Math.abs(end.y - aboveY);
            int distBelow = Math.abs(start.y - belowY) + Math.abs(end.y - belowY);
            int distBetween = betweenY >= 0 ? Math.abs(start.y - betweenY) + Math.abs(end.y - betweenY) : Integer.MAX_VALUE;

            if (distBetween <= distAbove && distBetween <= distBelow) {
                routeY = betweenY;
            } else if (distAbove <= distBelow) {
                routeY = aboveY;
            } else {
                routeY = belowY;
            }

            // Build a 5-segment path: out horizontally, up/down, across, down/up, in horizontally
            path.add(new Point(exitX, start.y));
            path.add(new Point(exitX, routeY));
            path.add(new Point(entryX, routeY));
            path.add(new Point(entryX, end.y));
        }

        path.add(end);
        return path;
    }

    /**
     * Draw a smooth bezier curve from start to end.
     */
    private void drawRoutedPath(GC gc, java.util.List<Point> path, PipelineNode sourceNode, PipelineNode targetNode) {
        if (path.size() < 2) return;

        Point start = path.get(0);
        Point end = path.get(path.size() - 1);

        int[] cp = calculateBezierControlPoints(start, end, sourceNode, targetNode);

        Path swtPath = new Path(gc.getDevice());
        try {
            swtPath.moveTo(start.x, start.y);
            swtPath.cubicTo(cp[0], cp[1], cp[2], cp[3], end.x, end.y);
            gc.drawPath(swtPath);
        } finally {
            swtPath.dispose();
        }
    }

    /**
     * Overload for cases without node info (dangling connections, etc.)
     */
    private void drawRoutedPath(GC gc, java.util.List<Point> path) {
        drawRoutedPath(gc, path, null, null);
    }

    /**
     * Calculate bezier control points for a smooth curve from start to end.
     * Returns [cx1, cy1, cx2, cy2].
     */
    private int[] calculateBezierControlPoints(Point start, Point end, PipelineNode sourceNode, PipelineNode targetNode) {
        int cx1, cy1, cx2, cy2;
        int dx = end.x - start.x;
        int dy = end.y - start.y;

        // If nearly horizontal (small Y difference), draw straight
        if (Math.abs(dy) < 10) {
            cx1 = start.x + dx / 3;
            cy1 = start.y + dy / 3;
            cx2 = start.x + 2 * dx / 3;
            cy2 = start.y + 2 * dy / 3;
        } else {
            // Has vertical offset - use S-curve with horizontal tangents
            int controlDist = Math.max(40, Math.abs(dx) / 2);
            cx1 = start.x + controlDist;
            cy1 = start.y;
            cx2 = end.x - controlDist;
            cy2 = end.y;
        }

        return new int[] { cx1, cy1, cx2, cy2 };
    }

    /**
     * Get the midpoint of a bezier curve (for label placement).
     * Uses the bezier formula at t=0.5.
     */
    private Point getPathMidpoint(java.util.List<Point> path, PipelineNode sourceNode, PipelineNode targetNode) {
        if (path.size() < 2) return path.get(0);

        Point start = path.get(0);
        Point end = path.get(path.size() - 1);

        // Get the same control points used for drawing
        int[] cp = calculateBezierControlPoints(start, end, sourceNode, targetNode);
        int cx1 = cp[0], cy1 = cp[1], cx2 = cp[2], cy2 = cp[3];

        // Cubic bezier at t=0.5: B(0.5) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        // At t=0.5: B(0.5) = 0.125*P0 + 0.375*P1 + 0.375*P2 + 0.125*P3
        double t = 0.5;
        double mt = 1 - t;
        double mt2 = mt * mt;
        double mt3 = mt2 * mt;
        double t2 = t * t;
        double t3 = t2 * t;

        int midX = (int) (mt3 * start.x + 3 * mt2 * t * cx1 + 3 * mt * t2 * cx2 + t3 * end.x);
        int midY = (int) (mt3 * start.y + 3 * mt2 * t * cy1 + 3 * mt * t2 * cy2 + t3 * end.y);

        return new Point(midX, midY);
    }

    /**
     * Overload for cases without node info.
     */
    private Point getPathMidpoint(java.util.List<Point> path) {
        return getPathMidpoint(path, null, null);
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
                if (node.hasDualInput()) {
                    Point inputPoint2 = node.getInputPoint2();
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
                if (inputPoint == null) continue;  // SourceNodes have no input
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
                // Check all output points for multi-output nodes
                int numOutputs = node.getOutputCount();
                for (int outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
                    Point outputPoint = node.getOutputPoint(outputIdx);
                    if (outputPoint == null) continue;
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
                        // Check if there's a connection from this specific output
                        Connection connToRemove = null;
                        for (Connection conn : connections) {
                            if (conn.source == node && conn.outputIndex == outputIdx) {
                                connToRemove = conn;
                                break;
                            }
                        }
                        if (connToRemove != null) {
                            // Yank off the connection - remove it and start dragging from the target
                            connectionTarget = connToRemove.target;
                            targetInputIndex = connToRemove.inputIndex;
                            connectionEndPoint = clickPoint;
                            connections.remove(connToRemove);
                            canvas.redraw();
                            return;
                        }
                        // Check if there's a dangling connection from this specific output
                        // If so, we should yank it to create a FreeConnection
                        DanglingConnection danglingToYank = null;
                        for (DanglingConnection dangling : danglingConnections) {
                            if (dangling.source == node && dangling.outputIndex == outputIdx) {
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
                        connectionSourceOutputIndex = outputIdx;
                        connectionEndPoint = clickPoint;
                        canvas.redraw();
                        return;
                    }
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
                    if (node.hasDualInput()) {
                        Point inputPoint2 = node.getInputPoint2();
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
                    if (inputPoint == null) continue;  // SourceNodes have no input
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
                // Create a new connection with the outputIndex
                connections.add(new Connection(connectionSource, targetNode, inputIdx, connectionSourceOutputIndex));
                markDirty();
            } else if (connectionEndPoint != null) {
                // Create a dangling connection with the outputIndex
                danglingConnections.add(new DanglingConnection(connectionSource, connectionSourceOutputIndex, connectionEndPoint));
                markDirty();
            } else {
            }

            connectionSource = null;
            connectionSourceOutputIndex = 0;
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
            int sourceOutputIndex = 0;

            // Check if dropped on an output point (check all outputs for multi-output nodes)
            for (PipelineNode node : nodes) {
                if (node != connectionTarget) {
                    int radius = 8;
                    int outputCount = node.getOutputCount();
                    for (int outIdx = 0; outIdx < outputCount; outIdx++) {
                        Point outputPoint = node.getOutputPoint(outIdx);
                        if (outputPoint != null) {
                            double dist = Math.sqrt(Math.pow(clickPoint.x - outputPoint.x, 2) +
                                                   Math.pow(clickPoint.y - outputPoint.y, 2));
                            if (dist <= radius) {
                                sourceNode = node;
                                sourceOutputIndex = outIdx;
                                connected = true;
                                break;
                            }
                        }
                    }
                    if (connected) break;
                }
            }

            // If not on output point, check if on node body as fallback (use output 0)
            if (!connected) {
                for (PipelineNode node : nodes) {
                    if (node != connectionTarget && node.containsPoint(clickPoint)) {
                        sourceNode = node;
                        sourceOutputIndex = 0;
                        connected = true;
                        break;
                    }
                }
            }

            if (connected && sourceNode != null) {
                // Create a new connection with the correct input and output index
                connections.add(new Connection(sourceNode, connectionTarget, targetInputIndex, sourceOutputIndex));
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
                // Dragging the source end - check if dropped on output point (check all for multi-output)
                PipelineNode sourceNode = null;
                int sourceOutputIndex = 0;
                for (PipelineNode node : nodes) {
                    int outputCount = node.getOutputCount();
                    for (int outIdx = 0; outIdx < outputCount; outIdx++) {
                        Point outputPoint = node.getOutputPoint(outIdx);
                        if (outputPoint != null) {
                            double dist = Math.sqrt(Math.pow(clickPoint.x - outputPoint.x, 2) +
                                                   Math.pow(clickPoint.y - outputPoint.y, 2));
                            if (dist <= radius) {
                                sourceNode = node;
                                sourceOutputIndex = outIdx;
                                connected = true;
                                break;
                            }
                        }
                    }
                    if (connected) break;
                }

                if (connected && sourceNode != null) {
                    // Connected to output point - create DanglingConnection with output index
                    danglingConnections.add(new DanglingConnection(sourceNode, sourceOutputIndex, freeConnectionFixedEnd));
                } else {
                    // Not connected - create FreeConnection at current position
                    freeConnections.add(new FreeConnection(connectionEndPoint, freeConnectionFixedEnd));
                }
            } else {
                // Dragging the target end - check if dropped on input point
                PipelineNode targetNode = null;
                for (PipelineNode node : nodes) {
                    Point inputPoint = node.getInputPoint();
                    if (inputPoint == null) continue;  // SourceNodes have no input
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

        // Check for tooltip on connection points
        updateConnectionTooltip(canvasX, canvasY);

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

    /**
     * Update canvas tooltip based on mouse position over connection points.
     */
    private void updateConnectionTooltip(int canvasX, int canvasY) {
        String tooltip = null;

        // Check all nodes for connection point hover
        for (PipelineNode node : nodes) {
            // Check output points
            int outputIndex = node.getOutputIndexNear(canvasX, canvasY);
            if (outputIndex >= 0) {
                tooltip = node.getOutputTooltip(outputIndex);
                break;
            }

            // Check primary input point
            if (node.isNearInputPoint(canvasX, canvasY)) {
                tooltip = node.getInputTooltip();
                break;
            }

            // Check secondary input point (dual-input nodes)
            if (node.isNearInputPoint2(canvasX, canvasY)) {
                tooltip = node.getInput2Tooltip();
                break;
            }
        }

        // Update canvas tooltip
        String currentTooltip = canvas.getToolTipText();
        if (tooltip == null) {
            if (currentTooltip != null) {
                canvas.setToolTipText(null);
            }
        } else {
            if (!tooltip.equals(currentTooltip)) {
                canvas.setToolTipText(tooltip);
            }
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
