package com.ttennebkram.pipeline.fx;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebView;
import javafx.scene.web.WebEngine;
import javafx.stage.Modality;
import javafx.stage.Stage;

import netscape.javascript.JSObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * A popup browser window for displaying help documentation.
 * Handles loading HTML files from resources (inside JAR or filesystem)
 * and intercepts internal links to navigate within the help system.
 */
public class FXHelpBrowser {

    private Stage stage;
    private WebView webView;
    private WebEngine webEngine;
    private Button backButton;
    private List<String> history = new ArrayList<>();
    private int historyIndex = -1;
    private boolean navigatingFromHistory = false;

    // Default window size
    private static final int DEFAULT_WIDTH = 700;
    private static final int DEFAULT_HEIGHT = 500;

    // JavaScript bridge for handling link clicks
    private LinkClickBridge linkClickBridge;

    /**
     * Bridge class to receive link click events from JavaScript.
     * Must be public for JavaScript to access it.
     */
    public class LinkClickBridge {
        public void handleLink(String href) {
            if (href == null || href.isEmpty()) return;

            // Handle external URLs
            if (href.startsWith("http://") || href.startsWith("https://")) {
                openInSystemBrowser(href);
                return;
            }

            // Handle internal doc links
            String docPath = href;
            // Remove leading slash if present
            if (docPath.startsWith("/")) {
                docPath = docPath.substring(1);
            }
            // If path doesn't start with doc/, might need to resolve relative to current doc
            if (!docPath.startsWith("doc/") && historyIndex >= 0) {
                // Get the current doc directory
                String currentDoc = history.get(historyIndex);
                int lastSlash = currentDoc.lastIndexOf('/');
                if (lastSlash > 0) {
                    String baseDir = currentDoc.substring(0, lastSlash + 1);
                    docPath = baseDir + docPath;
                }
            }
            navigate(docPath);
        }
    }

    /**
     * Open a help browser window showing the specified doc path.
     * @param parent Parent stage (for positioning)
     * @param docPath Path relative to resources, e.g., "doc/opencv/GaussianBlur.html"
     */
    public static void open(Stage parent, String docPath) {
        try {
            FXHelpBrowser helpBrowser = new FXHelpBrowser(parent);
            helpBrowser.navigate(docPath);
            helpBrowser.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Open the About page with version information substituted.
     * @param parent Parent stage (for positioning)
     */
    public static void openAbout(Stage parent) {
        try {
            FXHelpBrowser helpBrowser = new FXHelpBrowser(parent);
            helpBrowser.navigateAbout();
            helpBrowser.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Open the README documentation.
     * @param parent Parent stage (for positioning)
     */
    public static void openReadme(Stage parent) {
        open(parent, "README.md");
    }

    /**
     * Open help for a specific node type.
     * @param parent Parent stage
     * @param nodeType The node type name to show help for
     */
    public static void openForNodeType(Stage parent, String nodeType) {
        String docPath = getDocPathForNodeType(nodeType);
        if (docPath != null) {
            open(parent, docPath);
        } else {
            open(parent, "doc/no-help.html");
        }
    }

    /**
     * Open a searchable help browser showing all documented node types.
     * @param parent Parent stage
     */
    public static void openSearch(Stage parent) {
        Stage searchStage = new Stage();
        searchStage.setTitle("Search OpenCV Pipeline Help");
        searchStage.initOwner(parent);
        searchStage.initModality(Modality.NONE);

        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        // Search field
        TextField searchField = new TextField();
        searchField.setPromptText("Type to search nodes...");

        // Get all documented node types
        Map<String, String> allNodes = getAllDocumentedNodes();
        List<String> allNodeNames = new ArrayList<>(allNodes.keySet());

        // Results list
        ListView<String> resultsList = new ListView<>();
        resultsList.getItems().addAll(allNodeNames);
        VBox.setVgrow(resultsList, Priority.ALWAYS);

        // Filter as user types
        searchField.textProperty().addListener((obs, oldVal, newVal) -> {
            resultsList.getItems().clear();
            String filter = newVal.toLowerCase();
            for (String name : allNodeNames) {
                if (name.toLowerCase().contains(filter)) {
                    resultsList.getItems().add(name);
                }
            }
        });

        // Open help on single click (more intuitive)
        resultsList.setOnMouseClicked(e -> {
            String selected = resultsList.getSelectionModel().getSelectedItem();
            if (selected != null) {
                String docPath = allNodes.get(selected);
                if (docPath != null) {
                    open(parent, docPath);
                }
            }
        });

        resultsList.setOnKeyPressed(e -> {
            if (e.getCode() == javafx.scene.input.KeyCode.ENTER) {
                String selected = resultsList.getSelectionModel().getSelectedItem();
                if (selected != null) {
                    String docPath = allNodes.get(selected);
                    if (docPath != null) {
                        open(parent, docPath);
                    }
                }
            }
        });

        VBox content = new VBox(10);
        content.getChildren().addAll(searchField, resultsList);

        root.setCenter(content);

        Scene scene = new Scene(root, 350, 400);
        searchStage.setScene(scene);

        // Position relative to parent
        if (parent != null) {
            searchStage.setX(parent.getX() + 50);
            searchStage.setY(parent.getY() + 50);
        }

        searchStage.show();
        searchField.requestFocus();
    }

    /**
     * Get a map of display names to doc paths for all documented nodes.
     */
    private static Map<String, String> getAllDocumentedNodes() {
        Map<String, String> nodes = new LinkedHashMap<>();

        // General documentation (at top of list)
        nodes.put("README - Full Documentation", "README.md");

        // Blur nodes
        nodes.put("Gaussian Blur", "doc/opencv/GaussianBlur.html");
        nodes.put("Box Blur", "doc/opencv/blur.html");
        nodes.put("Median Blur", "doc/opencv/medianBlur.html");
        nodes.put("Bilateral Blur", "doc/opencv/bilateralFilter.html");

        // Edge detection
        nodes.put("Canny Edge", "doc/opencv/Canny.html");
        nodes.put("Sobel", "doc/opencv/Sobel.html");
        nodes.put("Laplacian", "doc/opencv/Laplacian.html");
        nodes.put("Scharr", "doc/opencv/Scharr.html");

        // Threshold
        nodes.put("Threshold", "doc/opencv/threshold.html");
        nodes.put("Adaptive Threshold", "doc/opencv/adaptiveThreshold.html");

        // Morphology
        nodes.put("Dilate", "doc/opencv/dilate.html");
        nodes.put("Erode", "doc/opencv/erode.html");
        nodes.put("Morphology Ex", "doc/opencv/morphologyEx.html");

        // Color/conversion
        nodes.put("Grayscale", "doc/opencv/cvtColor.html");
        nodes.put("Color In Range", "doc/opencv/inRange.html");

        // Bitwise
        nodes.put("Bitwise And", "doc/opencv/bitwise_and.html");
        nodes.put("Bitwise Or", "doc/opencv/bitwise_or.html");
        nodes.put("Bitwise Xor", "doc/opencv/bitwise_xor.html");
        nodes.put("Bitwise Not / Invert", "doc/opencv/bitwise_not.html");

        // Arithmetic
        nodes.put("Add (Clamp)", "doc/opencv/add.html");
        nodes.put("Subtract (Clamp)", "doc/opencv/subtract.html");
        nodes.put("Add Weighted", "doc/opencv/addWeighted.html");

        // Features
        nodes.put("Hough Lines", "doc/opencv/HoughLines.html");
        nodes.put("Hough Circles", "doc/opencv/HoughCircles.html");
        nodes.put("Contours", "doc/opencv/findContours.html");
        nodes.put("Harris Corners", "doc/opencv/cornerHarris.html");
        nodes.put("Shi-Tomasi Corners", "doc/opencv/goodFeaturesToTrack.html");
        nodes.put("ORB Features", "doc/opencv/ORB.html");
        nodes.put("SIFT Features", "doc/opencv/SIFT.html");
        nodes.put("Blob Detector", "doc/opencv/SimpleBlobDetector.html");

        // Drawing
        nodes.put("Circle", "doc/opencv/circle.html");
        nodes.put("Ellipse", "doc/opencv/ellipse.html");
        nodes.put("Line", "doc/opencv/line.html");
        nodes.put("Arrow", "doc/opencv/arrowedLine.html");
        nodes.put("Rectangle", "doc/opencv/rectangle.html");
        nodes.put("Text", "doc/opencv/putText.html");

        // Other OpenCV
        nodes.put("Histogram", "doc/opencv/calcHist.html");
        nodes.put("CLAHE", "doc/opencv/CLAHE.html");
        nodes.put("Warp Affine", "doc/opencv/warpAffine.html");
        nodes.put("Crop", "doc/opencv/Mat_submat.html");
        nodes.put("Gain", "doc/opencv/convertTo.html");
        nodes.put("Match Template", "doc/opencv/matchTemplate.html");
        nodes.put("Connected Components", "doc/opencv/connectedComponents.html");
        nodes.put("Filter 2D", "doc/opencv/filter2D.html");
        nodes.put("Mean Shift Blur", "doc/opencv/pyrMeanShiftFiltering.html");
        nodes.put("Resize", "doc/opencv/resize.html");

        // FFT nodes
        nodes.put("FFT Filters (Low/High Pass)", "doc/opencv/dft.html");

        // Custom nodes
        nodes.put("Bit Planes", "doc/nodes/BitPlanesNode.html");
        nodes.put("Blur Highpass", "doc/nodes/BlurHighpassNode.html");
        nodes.put("Divide By Background", "doc/nodes/DivideByBackgroundNode.html");
        nodes.put("Clone", "doc/nodes/CloneNode.html");
        nodes.put("Container", "doc/nodes/ContainerNode.html");
        nodes.put("Monitor", "doc/nodes/MonitorNode.html");

        // Source nodes
        nodes.put("File Source", "doc/nodes/FileSourceNode.html");
        nodes.put("Webcam Source", "doc/nodes/WebcamSourceNode.html");
        nodes.put("Blank Source", "doc/nodes/BlankSourceNode.html");

        // Container I/O
        nodes.put("Sub-pipeline Boundary Input", "doc/nodes/ContainerInputNode.html");
        nodes.put("Sub-pipeline Boundary Output", "doc/nodes/ContainerOutputNode.html");
        nodes.put("Is-Nested Input", "doc/nodes/IsNestedInputNode.html");
        nodes.put("Is-Nested Output", "doc/nodes/IsNestedOutputNode.html");

        return nodes;
    }

    /**
     * Get the documentation path for a node type.
     */
    public static String getDocPathForNodeType(String nodeType) {
        switch (nodeType) {
            // Blur nodes
            case "GaussianBlur":
                return "doc/opencv/GaussianBlur.html";
            case "BoxBlur":
                return "doc/opencv/blur.html";
            case "MedianBlur":
                return "doc/opencv/medianBlur.html";
            case "BilateralFilter":
                return "doc/opencv/bilateralFilter.html";

            // Edge detection
            case "CannyEdge":
                return "doc/opencv/Canny.html";
            case "Sobel":
                return "doc/opencv/Sobel.html";
            case "Laplacian":
                return "doc/opencv/Laplacian.html";
            case "Scharr":
                return "doc/opencv/Scharr.html";

            // Threshold
            case "Threshold":
                return "doc/opencv/threshold.html";
            case "AdaptiveThreshold":
                return "doc/opencv/adaptiveThreshold.html";

            // Morphology
            case "Dilate":
                return "doc/opencv/dilate.html";
            case "Erode":
                return "doc/opencv/erode.html";
            case "MorphOpen":
            case "MorphClose":
            case "MorphologyEx":
                return "doc/opencv/morphologyEx.html";

            // Color/conversion
            case "Grayscale":
                return "doc/opencv/cvtColor.html";
            case "ColorInRange":
                return "doc/opencv/inRange.html";

            // Bitwise
            case "BitwiseAnd":
                return "doc/opencv/bitwise_and.html";
            case "BitwiseOr":
                return "doc/opencv/bitwise_or.html";
            case "BitwiseXor":
                return "doc/opencv/bitwise_xor.html";
            case "BitwiseNot":
                return "doc/opencv/bitwise_not.html";

            // Arithmetic
            case "AddClamp":
                return "doc/opencv/add.html";
            case "SubtractClamp":
                return "doc/opencv/subtract.html";
            case "AddWeighted":
                return "doc/opencv/addWeighted.html";

            // Features
            case "HoughLines":
                return "doc/opencv/HoughLines.html";
            case "HoughCircles":
                return "doc/opencv/HoughCircles.html";
            case "Contours":
                return "doc/opencv/findContours.html";
            case "HarrisCorners":
                return "doc/opencv/cornerHarris.html";
            case "ShiTomasiCorners":
                return "doc/opencv/goodFeaturesToTrack.html";
            case "ORBFeatures":
                return "doc/opencv/ORB.html";
            case "SIFTFeatures":
                return "doc/opencv/SIFT.html";
            case "BlobDetector":
                return "doc/opencv/SimpleBlobDetector.html";

            // Drawing
            case "Circle":
                return "doc/opencv/circle.html";
            case "Ellipse":
                return "doc/opencv/ellipse.html";
            case "Line":
                return "doc/opencv/line.html";
            case "Arrow":
                return "doc/opencv/arrowedLine.html";
            case "Rectangle":
                return "doc/opencv/rectangle.html";
            case "Text":
                return "doc/opencv/putText.html";

            // Other OpenCV
            case "Histogram":
                return "doc/opencv/calcHist.html";
            case "CLAHE":
                return "doc/opencv/CLAHE.html";
            case "WarpAffine":
                return "doc/opencv/warpAffine.html";
            case "Crop":
                return "doc/opencv/Mat_submat.html";
            case "Gain":
                return "doc/opencv/convertTo.html";
            case "Invert":
                return "doc/opencv/bitwise_not.html";
            case "MatchTemplate":
                return "doc/opencv/matchTemplate.html";
            case "ConnectedComponents":
                return "doc/opencv/connectedComponents.html";
            case "Filter2D":
                return "doc/opencv/filter2D.html";
            case "MeanShiftFilter":
                return "doc/opencv/pyrMeanShiftFiltering.html";

            // FFT nodes
            case "FFTLowPassFilter":
            case "FFTLowPass4":
            case "FFTHighPassFilter":
            case "FFTHighPass4":
                return "doc/opencv/dft.html";

            // Bit planes
            case "BitPlanesGrayscale":
            case "BitPlanesColor":
                return "doc/nodes/BitPlanesNode.html";

            // Custom/utility nodes
            case "Clone":
                return "doc/nodes/CloneNode.html";
            case "Container":
                return "doc/nodes/ContainerNode.html";
            case "Monitor":
                return "doc/nodes/MonitorNode.html";

            // Source nodes
            case "FileSource":
                return "doc/nodes/FileSourceNode.html";
            case "WebcamSource":
                return "doc/nodes/WebcamSourceNode.html";
            case "BlankSource":
                return "doc/nodes/BlankSourceNode.html";

            // Container boundary nodes
            case "ContainerInput":
                return "doc/nodes/ContainerInputNode.html";
            case "ContainerOutput":
                return "doc/nodes/ContainerOutputNode.html";

            // Is-Nested routing nodes
            case "IsNestedInput":
                return "doc/nodes/IsNestedInputNode.html";
            case "IsNestedOutput":
                return "doc/nodes/IsNestedOutputNode.html";

            // Transform nodes
            case "Resize":
                return "doc/opencv/resize.html";

            // Other custom nodes
            case "BlurHighpass":
                return "doc/nodes/BlurHighpassNode.html";
            case "DivideByBackground":
                return "doc/nodes/DivideByBackgroundNode.html";

            default:
                return null;
        }
    }

    /**
     * Check if help is available for a node type.
     * Simply checks if there's a mapping in getDocPathForNodeType().
     */
    public static boolean hasHelp(String nodeType) {
        return getDocPathForNodeType(nodeType) != null;
    }

    private FXHelpBrowser(Stage parent) {
        stage = new Stage();
        stage.setTitle("Help");
        stage.initOwner(parent);
        stage.initModality(Modality.NONE);

        // Initialize the JavaScript bridge
        linkClickBridge = new LinkClickBridge();

        BorderPane root = new BorderPane();

        // Toolbar with back button
        HBox toolbar = new HBox(10);
        toolbar.setPadding(new Insets(5));

        backButton = new Button("\u2190 Back");
        backButton.setDisable(true);
        backButton.setOnAction(e -> goBack());
        toolbar.getChildren().add(backButton);

        root.setTop(toolbar);

        // WebView for HTML content
        webView = new WebView();
        webEngine = webView.getEngine();

        // Setup JavaScript bridge when document loads
        webEngine.getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
            if (newState == javafx.concurrent.Worker.State.SUCCEEDED) {
                setupJavaScriptBridge();
            }
        });

        // Handle link clicks (fallback for location changes)
        webEngine.locationProperty().addListener((obs, oldLoc, newLoc) -> {
            if (newLoc != null && !newLoc.isEmpty() && !newLoc.equals("about:blank")) {
                handleLocationChange(newLoc);
            }
        });

        root.setCenter(webView);

        Scene scene = new Scene(root, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        stage.setScene(scene);

        // Position relative to parent
        if (parent != null) {
            stage.setX(parent.getX() + (parent.getWidth() - DEFAULT_WIDTH) / 2);
            stage.setY(parent.getY() + (parent.getHeight() - DEFAULT_HEIGHT) / 2);
        }
    }

    /**
     * Set up the JavaScript bridge to intercept link clicks.
     */
    private void setupJavaScriptBridge() {
        try {
            JSObject window = (JSObject) webEngine.executeScript("window");
            window.setMember("javaLinkHandler", linkClickBridge);

            // Inject JavaScript to intercept all link clicks
            webEngine.executeScript(
                "document.addEventListener('click', function(e) {" +
                "  var target = e.target;" +
                "  while (target && target.tagName !== 'A') {" +
                "    target = target.parentElement;" +
                "  }" +
                "  if (target && target.href) {" +
                "    var href = target.getAttribute('href');" +
                "    if (href && !href.startsWith('#')) {" +
                "      e.preventDefault();" +
                "      javaLinkHandler.handleLink(href);" +
                "    }" +
                "  }" +
                "}, true);"
            );
        } catch (Exception e) {
            System.err.println("Failed to setup JavaScript bridge: " + e.getMessage());
        }
    }

    private void handleLocationChange(String location) {
        // Handle external URLs
        if (location.startsWith("http://") || location.startsWith("https://")) {
            openInSystemBrowser(location);
            // Navigate back in WebView
            if (historyIndex >= 0) {
                navigateToHistoryItem(historyIndex);
            }
            return;
        }

        // Handle internal doc links
        // When content is loaded via loadContent(), relative links may appear in various formats:
        // - "doc/types/Mat.html" (original href)
        // - "about:blank" with the href not captured
        // - URL-encoded or with file:// prefix
        String docPath = null;

        if (location.contains("doc/")) {
            int docIndex = location.indexOf("doc/");
            docPath = location.substring(docIndex);
            // Remove any query params or hash
            if (docPath.contains("?")) {
                docPath = docPath.substring(0, docPath.indexOf("?"));
            }
            if (docPath.contains("#")) {
                docPath = docPath.substring(0, docPath.indexOf("#"));
            }
        } else if (location.startsWith("doc/")) {
            docPath = location;
        }

        if (docPath != null) {
            navigate(docPath);
        }
    }

    private void navigate(String docPath) {
        String content = loadResource(docPath);
        if (content != null) {
            // Convert markdown to HTML if needed
            String html;
            if (docPath.endsWith(".md")) {
                html = convertMarkdownToHtml(content);
            } else {
                html = content;
            }

            if (!navigatingFromHistory) {
                // Remove forward history
                while (history.size() > historyIndex + 1) {
                    history.remove(history.size() - 1);
                }
                history.add(docPath);
                historyIndex = history.size() - 1;
            }
            updateBackButton();

            webEngine.loadContent(html);

            // Update title
            String title = extractTitle(html);
            if (title != null) {
                stage.setTitle("Help - " + title);
            }
        } else {
            webEngine.loadContent(getNotFoundHtml(docPath));
        }
    }

    /**
     * Navigate to the About page with version info substituted.
     */
    private void navigateAbout() {
        String html = loadResource("doc/about.html");
        if (html != null) {
            // Substitute the OpenCV version
            html = html.replace("{{OPENCV_VERSION}}", org.opencv.core.Core.VERSION);

            if (!navigatingFromHistory) {
                while (history.size() > historyIndex + 1) {
                    history.remove(history.size() - 1);
                }
                history.add("doc/about.html");
                historyIndex = history.size() - 1;
            }
            updateBackButton();

            webEngine.loadContent(html);
            stage.setTitle("About OpenCV Pipeline Editor");
        } else {
            webEngine.loadContent(getNotFoundHtml("doc/about.html"));
        }
    }

    /**
     * Convert basic markdown to HTML for display.
     * Handles headers, lists, code blocks, links, bold, and basic formatting.
     */
    private String convertMarkdownToHtml(String markdown) {
        StringBuilder html = new StringBuilder();
        html.append("<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><title>README</title>");
        html.append("<style>");
        html.append("body { font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; margin: 20px; line-height: 1.6; }");
        html.append("h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }");
        html.append("h2 { color: #34495e; margin-top: 25px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }");
        html.append("h3 { color: #555; margin-top: 20px; }");
        html.append("code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }");
        html.append("pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }");
        html.append("pre code { background: none; padding: 0; }");
        html.append("a { color: #3498db; }");
        html.append("table { border-collapse: collapse; margin: 15px 0; }");
        html.append("th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }");
        html.append("th { background: #f4f4f4; }");
        html.append("ul, ol { padding-left: 25px; }");
        html.append("li { margin: 5px 0; }");
        html.append("blockquote { border-left: 4px solid #3498db; margin: 15px 0; padding-left: 15px; color: #666; }");
        html.append("img { max-width: 100%; height: auto; }");
        html.append("</style></head><body>");

        String[] lines = markdown.split("\n");
        boolean inCodeBlock = false;
        boolean inList = false;
        boolean inTable = false;
        StringBuilder codeBlock = new StringBuilder();

        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];

            // Code blocks
            if (line.startsWith("```")) {
                if (inCodeBlock) {
                    html.append("<pre><code>").append(escapeHtml(codeBlock.toString())).append("</code></pre>\n");
                    codeBlock = new StringBuilder();
                    inCodeBlock = false;
                } else {
                    if (inList) {
                        html.append("</ul>\n");
                        inList = false;
                    }
                    inCodeBlock = true;
                }
                continue;
            }

            if (inCodeBlock) {
                codeBlock.append(line).append("\n");
                continue;
            }

            // Tables
            if (line.startsWith("|") && line.endsWith("|")) {
                if (!inTable) {
                    if (inList) {
                        html.append("</ul>\n");
                        inList = false;
                    }
                    html.append("<table>\n");
                    inTable = true;
                }
                // Skip separator lines like |---|---|
                if (line.matches("\\|[-:\\s|]+\\|")) {
                    continue;
                }
                String[] cells = line.split("\\|");
                html.append("<tr>");
                for (int c = 1; c < cells.length - 1; c++) { // Skip empty first/last from split
                    String cell = cells[c].trim();
                    cell = processInlineMarkdown(cell);
                    // First row is header
                    if (i > 0 && lines[i-1].startsWith("|") && !lines[i-1].matches("\\|[-:\\s|]+\\|")) {
                        html.append("<td>").append(cell).append("</td>");
                    } else {
                        html.append("<th>").append(cell).append("</th>");
                    }
                }
                html.append("</tr>\n");
                continue;
            } else if (inTable) {
                html.append("</table>\n");
                inTable = false;
            }

            // Headers
            if (line.startsWith("### ")) {
                if (inList) { html.append("</ul>\n"); inList = false; }
                html.append("<h3>").append(processInlineMarkdown(line.substring(4))).append("</h3>\n");
                continue;
            }
            if (line.startsWith("## ")) {
                if (inList) { html.append("</ul>\n"); inList = false; }
                html.append("<h2>").append(processInlineMarkdown(line.substring(3))).append("</h2>\n");
                continue;
            }
            if (line.startsWith("# ")) {
                if (inList) { html.append("</ul>\n"); inList = false; }
                html.append("<h1>").append(processInlineMarkdown(line.substring(2))).append("</h1>\n");
                continue;
            }

            // List items
            if (line.startsWith("- ") || line.startsWith("* ")) {
                if (!inList) {
                    html.append("<ul>\n");
                    inList = true;
                }
                html.append("<li>").append(processInlineMarkdown(line.substring(2))).append("</li>\n");
                continue;
            }

            // Numbered lists
            if (line.matches("^\\d+\\.\\s.*")) {
                if (!inList) {
                    html.append("<ul>\n");
                    inList = true;
                }
                html.append("<li>").append(processInlineMarkdown(line.replaceFirst("^\\d+\\.\\s", ""))).append("</li>\n");
                continue;
            }

            // End list if non-list line
            if (inList && !line.trim().isEmpty()) {
                html.append("</ul>\n");
                inList = false;
            }

            // Empty lines
            if (line.trim().isEmpty()) {
                if (inList) {
                    html.append("</ul>\n");
                    inList = false;
                }
                continue;
            }

            // Regular paragraph
            html.append("<p>").append(processInlineMarkdown(line)).append("</p>\n");
        }

        // Close any open tags
        if (inList) html.append("</ul>\n");
        if (inTable) html.append("</table>\n");
        if (inCodeBlock) html.append("<pre><code>").append(escapeHtml(codeBlock.toString())).append("</code></pre>\n");

        html.append("</body></html>");
        return html.toString();
    }

    /**
     * Process inline markdown: bold, italic, code, links.
     */
    private String processInlineMarkdown(String text) {
        // Escape HTML first (but preserve our conversions)
        text = escapeHtml(text);

        // Inline code (must come before bold/italic to avoid conflicts)
        text = text.replaceAll("`([^`]+)`", "<code>$1</code>");

        // Bold **text** or __text__
        text = text.replaceAll("\\*\\*([^*]+)\\*\\*", "<strong>$1</strong>");
        text = text.replaceAll("__([^_]+)__", "<strong>$1</strong>");

        // Links [text](url)
        text = text.replaceAll("\\[([^\\]]+)\\]\\(([^)]+)\\)", "<a href=\"$2\">$1</a>");

        // Images ![alt](url) - convert to just showing alt text since images may not load
        text = text.replaceAll("!\\[([^\\]]+)\\]\\(([^)]+)\\)", "<em>[Image: $1]</em>");

        return text;
    }

    /**
     * Escape HTML special characters.
     */
    private String escapeHtml(String text) {
        return text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("\"", "&quot;");
    }

    private void navigateToHistoryItem(int index) {
        if (index >= 0 && index < history.size()) {
            String html = loadResource(history.get(index));
            if (html != null) {
                webEngine.loadContent(html);
            }
        }
    }

    private void goBack() {
        if (historyIndex > 0) {
            historyIndex--;
            navigatingFromHistory = true;
            navigate(history.get(historyIndex));
            navigatingFromHistory = false;
            updateBackButton();
        }
    }

    private void updateBackButton() {
        backButton.setDisable(historyIndex <= 0);
    }

    private void show() {
        stage.show();
        stage.toFront();
    }

    private String loadResource(String path) {
        try (InputStream is = getClass().getResourceAsStream("/" + path)) {
            if (is == null) {
                return null;
            }
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(is, StandardCharsets.UTF_8))) {
                return reader.lines().collect(Collectors.joining("\n"));
            }
        } catch (Exception e) {
            System.err.println("Failed to load help resource: " + path + " - " + e.getMessage());
            return null;
        }
    }

    private String extractTitle(String html) {
        int start = html.indexOf("<title>");
        int end = html.indexOf("</title>");
        if (start >= 0 && end > start) {
            return html.substring(start + 7, end);
        }
        start = html.indexOf("<h1>");
        end = html.indexOf("</h1>");
        if (start >= 0 && end > start) {
            return html.substring(start + 4, end).replaceAll("<[^>]+>", "");
        }
        return null;
    }

    private String getNotFoundHtml(String docPath) {
        return "<!DOCTYPE html><html><head><title>Help Not Found</title>" +
               "<style>body { font-family: sans-serif; margin: 40px; text-align: center; }" +
               "h1 { color: #e74c3c; }</style></head>" +
               "<body><h1>Help Not Available</h1>" +
               "<p>No documentation found for: <code>" + docPath + "</code></p>" +
               "<p>This help page has not been created yet.</p></body></html>";
    }

    private static void openInSystemBrowser(String url) {
        try {
            java.awt.Desktop desktop = java.awt.Desktop.getDesktop();
            if (desktop.isSupported(java.awt.Desktop.Action.BROWSE)) {
                desktop.browse(new java.net.URI(url));
                return;
            }
        } catch (Exception e) {
            System.err.println("Failed to open URL: " + e.getMessage());
        }

        // macOS fallback
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("mac")) {
            try {
                Runtime.getRuntime().exec(new String[]{"open", url});
            } catch (Exception e) {
                System.err.println("Failed to open URL with 'open' command: " + e.getMessage());
            }
        }
    }
}
