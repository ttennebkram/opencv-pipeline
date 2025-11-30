package com.ttennebkram.pipeline.fx;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
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
import java.util.List;
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

            default:
                return null;
        }
    }

    /**
     * Check if help is available for a node type.
     */
    public static boolean hasHelp(String nodeType) {
        String docPath = getDocPathForNodeType(nodeType);
        if (docPath == null) {
            return false;
        }
        return FXHelpBrowser.class.getResourceAsStream("/" + docPath) != null;
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
        String html = loadResource(docPath);
        if (html != null) {
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
