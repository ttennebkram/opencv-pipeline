package com.ttennebkram.pipeline.ui;

import org.eclipse.swt.SWT;
import org.eclipse.swt.browser.Browser;
import org.eclipse.swt.browser.BrowserFunction;
import org.eclipse.swt.browser.LocationAdapter;
import org.eclipse.swt.browser.LocationEvent;
import org.eclipse.swt.graphics.Rectangle;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.program.Program;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;

import java.awt.Desktop;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A popup browser window for displaying help documentation.
 * Handles loading HTML files from resources (inside JAR or filesystem)
 * and intercepts internal links to navigate within the help system.
 */
public class HelpBrowser {

    private Shell shell;
    private Browser browser;
    private Button backButton;
    private List<String> history = new ArrayList<>();
    private int historyIndex = -1;
    private boolean navigatingFromHistory = false;

    // Default window size
    private static final int DEFAULT_WIDTH = 700;
    private static final int DEFAULT_HEIGHT = 500;

    /**
     * Open a help browser window showing the specified doc path.
     * @param parent Parent shell (for positioning)
     * @param docPath Path relative to resources, e.g., "doc/opencv/GaussianBlur.html"
     */
    public static void open(Shell parent, String docPath) {
        HelpBrowser helpBrowser = new HelpBrowser(parent);
        helpBrowser.navigate(docPath);
        helpBrowser.show();
    }

    /**
     * Open help for a specific node class.
     * Automatically determines the doc path based on the class.
     * @param parent Parent shell
     * @param nodeClass The node class to show help for
     */
    public static void openForNode(Shell parent, Class<?> nodeClass) {
        String docPath = getDocPathForClass(nodeClass);
        if (docPath != null) {
            open(parent, docPath);
        } else {
            // Show a "no help available" message
            open(parent, "doc/no-help.html");
        }
    }

    /**
     * Get the documentation path for a node class.
     * @param nodeClass The node class
     * @return The doc path, or null if no mapping exists
     */
    public static String getDocPathForClass(Class<?> nodeClass) {
        String className = nodeClass.getSimpleName();

        // Map node classes to their doc files
        // OpenCV-based nodes go to doc/opencv/
        // Custom nodes go to doc/nodes/

        switch (className) {
            // Blur nodes
            case "GaussianBlurNode":
                return "doc/opencv/GaussianBlur.html";
            case "BoxBlurNode":
                return "doc/opencv/blur.html";
            case "MedianBlurNode":
                return "doc/opencv/medianBlur.html";
            case "BilateralFilterNode":
                return "doc/opencv/bilateralFilter.html";

            // Edge detection
            case "CannyEdgeNode":
                return "doc/opencv/Canny.html";
            case "SobelNode":
                return "doc/opencv/Sobel.html";
            case "LaplacianNode":
                return "doc/opencv/Laplacian.html";
            case "ScharrNode":
                return "doc/opencv/Scharr.html";

            // Threshold
            case "ThresholdNode":
                return "doc/opencv/threshold.html";
            case "AdaptiveThresholdNode":
                return "doc/opencv/adaptiveThreshold.html";

            // Morphology
            case "DilateNode":
                return "doc/opencv/dilate.html";
            case "ErodeNode":
                return "doc/opencv/erode.html";
            case "MorphOpenNode":
                return "doc/opencv/morphologyEx.html";
            case "MorphCloseNode":
                return "doc/opencv/morphologyEx.html";
            case "MorphologyExNode":
                return "doc/opencv/morphologyEx.html";

            // Color/conversion
            case "GrayscaleNode":
                return "doc/opencv/cvtColor.html";
            case "ColorInRangeNode":
                return "doc/opencv/inRange.html";

            // Bitwise
            case "BitwiseAndNode":
                return "doc/opencv/bitwise_and.html";
            case "BitwiseOrNode":
                return "doc/opencv/bitwise_or.html";
            case "BitwiseXorNode":
                return "doc/opencv/bitwise_xor.html";
            case "BitwiseNotNode":
                return "doc/opencv/bitwise_not.html";

            // Arithmetic
            case "AddClampNode":
                return "doc/opencv/add.html";
            case "SubtractClampNode":
                return "doc/opencv/subtract.html";
            case "AddWeightedNode":
                return "doc/opencv/addWeighted.html";

            // Features
            case "HoughLinesNode":
                return "doc/opencv/HoughLines.html";
            case "HoughCirclesNode":
                return "doc/opencv/HoughCircles.html";
            case "ContoursNode":
                return "doc/opencv/findContours.html";
            case "HarrisCornersNode":
                return "doc/opencv/cornerHarris.html";
            case "ShiTomasiCornersNode":
                return "doc/opencv/goodFeaturesToTrack.html";
            case "ORBFeaturesNode":
                return "doc/opencv/ORB.html";
            case "SIFTFeaturesNode":
                return "doc/opencv/SIFT.html";
            case "BlobDetectorNode":
                return "doc/opencv/SimpleBlobDetector.html";

            // Drawing
            case "CircleNode":
                return "doc/opencv/circle.html";
            case "EllipseNode":
                return "doc/opencv/ellipse.html";
            case "LineNode":
                return "doc/opencv/line.html";
            case "ArrowNode":
                return "doc/opencv/arrowedLine.html";
            case "RectangleNode":
                return "doc/opencv/rectangle.html";
            case "TextNode":
                return "doc/opencv/putText.html";

            // Other OpenCV
            case "HistogramNode":
                return "doc/opencv/calcHist.html";
            case "CLAHENode":
                return "doc/opencv/CLAHE.html";
            case "WarpAffineNode":
                return "doc/opencv/warpAffine.html";
            case "CropNode":
                return "doc/opencv/Mat_submat.html";
            case "GainNode":
                return "doc/opencv/convertTo.html";
            case "InvertNode":
                return "doc/opencv/bitwise_not.html";
            case "MatchTemplateNode":
                return "doc/opencv/matchTemplate.html";
            case "ConnectedComponentsNode":
                return "doc/opencv/connectedComponents.html";
            case "Filter2DNode":
                return "doc/opencv/filter2D.html";
            case "MeanShiftFilterNode":
                return "doc/opencv/pyrMeanShiftFiltering.html";

            // FFT nodes
            case "FFTLowPassFilterNode":
            case "FFTLowPass4Node":
            case "FFTHighPassFilterNode":
            case "FFTHighPass4Node":
                return "doc/opencv/dft.html";

            // Bit planes
            case "BitPlanesGrayscaleNode":
            case "BitPlanesColorNode":
                return "doc/nodes/BitPlanesNode.html";

            // Custom/utility nodes
            case "CloneNode":
                return "doc/nodes/CloneNode.html";
            case "ContainerNode":
                return "doc/nodes/ContainerNode.html";
            case "MonitorNode":
                return "doc/nodes/MonitorNode.html";

            // Source nodes
            case "FileSourceNode":
                return "doc/nodes/FileSourceNode.html";
            case "WebcamSourceNode":
                return "doc/nodes/WebcamSourceNode.html";
            case "BlankSourceNode":
                return "doc/nodes/BlankSourceNode.html";

            default:
                return null;
        }
    }

    /**
     * Check if help is available for a node class.
     */
    public static boolean hasHelp(Class<?> nodeClass) {
        String docPath = getDocPathForClass(nodeClass);
        if (docPath == null) {
            return false;
        }
        // Check if the resource exists
        return HelpBrowser.class.getResourceAsStream("/" + docPath) != null;
    }

    private HelpBrowser(Shell parent) {
        shell = new Shell(parent, SWT.SHELL_TRIM | SWT.RESIZE);
        shell.setText("Help");
        shell.setSize(DEFAULT_WIDTH, DEFAULT_HEIGHT);

        GridLayout layout = new GridLayout(1, false);
        layout.marginWidth = 0;
        layout.marginHeight = 0;
        layout.verticalSpacing = 0;
        shell.setLayout(layout);

        // Center on parent
        Rectangle parentBounds = parent.getBounds();
        shell.setLocation(
            parentBounds.x + (parentBounds.width - DEFAULT_WIDTH) / 2,
            parentBounds.y + (parentBounds.height - DEFAULT_HEIGHT) / 2
        );

        // Toolbar with back button
        Composite toolbar = new Composite(shell, SWT.NONE);
        GridLayout toolbarLayout = new GridLayout(2, false);
        toolbarLayout.marginWidth = 5;
        toolbarLayout.marginHeight = 3;
        toolbar.setLayout(toolbarLayout);
        toolbar.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        backButton = new Button(toolbar, SWT.PUSH);
        backButton.setText("← Back");
        backButton.setEnabled(false);
        backButton.addListener(SWT.Selection, e -> goBack());

        browser = new Browser(shell, SWT.NONE);
        browser.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));

        // Create a Java function that JavaScript can call to handle link clicks
        new BrowserFunction(browser, "handleLink") {
            @Override
            public Object function(Object[] arguments) {
                System.out.println("JavaScript handleLink called with " + arguments.length + " args");
                if (arguments.length > 0 && arguments[0] instanceof String) {
                    String url = (String) arguments[0];
                    System.out.println("  URL: " + url);
                    handleLinkClick(url);
                }
                return null;
            }
        };

        // Intercept all navigation
        browser.addLocationListener(new LocationAdapter() {
            @Override
            public void changing(LocationEvent event) {
                String location = event.location;
                System.out.println("LocationListener.changing: " + location);

                // Ignore about:blank and initial loads
                if (location == null || location.equals("about:blank") || location.isEmpty()) {
                    return;
                }

                // Cancel all navigation - we handle everything ourselves
                event.doit = false;

                // Handle external URLs
                if (location.startsWith("http://") || location.startsWith("https://")) {
                    System.out.println("  -> Opening external URL");
                    openInSystemBrowser(location);
                    return;
                }

                // Handle internal doc links
                if (location.startsWith("doc/") || location.contains("/doc/")) {
                    String docPath = location;
                    int docIndex = location.indexOf("doc/");
                    if (docIndex >= 0) {
                        docPath = location.substring(docIndex);
                    }
                    System.out.println("  -> Navigating to internal doc: " + docPath);
                    navigate(docPath);
                    return;
                }

                System.out.println("  -> Unhandled URL format");
            }

            @Override
            public void changed(LocationEvent event) {
                System.out.println("LocationListener.changed: " + event.location);
            }
        });
    }

    /**
     * Handle a link click - either navigate internally or open in system browser.
     */
    private void handleLinkClick(String url) {
        // Handle external links - open in system browser
        if (url.startsWith("http://") || url.startsWith("https://")) {
            openInSystemBrowser(url);
            return;
        }

        // Handle internal doc links
        if (url.startsWith("doc/") || url.contains("/doc/")) {
            // Extract the doc path
            String docPath = url;
            int docIndex = url.indexOf("doc/");
            if (docIndex >= 0) {
                docPath = url.substring(docIndex);
            }
            navigate(docPath);
        }
    }

    private void navigate(String docPath) {
        String html = loadResource(docPath);
        if (html != null) {
            // Track history for back button
            if (!navigatingFromHistory) {
                // Remove any forward history
                while (history.size() > historyIndex + 1) {
                    history.remove(history.size() - 1);
                }
                history.add(docPath);
                historyIndex = history.size() - 1;
            }
            updateBackButton();

            // Transform links to use custom schemes we can intercept
            html = transformLinks(html);
            browser.setText(html);
            // Update window title based on page
            String title = extractTitle(html);
            if (title != null) {
                shell.setText("Help - " + title);
            }
        } else {
            browser.setText(transformLinks(getNotFoundHtml(docPath)));
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
        backButton.setEnabled(historyIndex > 0);
    }

    /**
     * No transformation needed - we'll intercept all navigation attempts.
     * Debug: print the full HTML being loaded.
     */
    private String transformLinks(String html) {
        // Debug: print the signature section
        int sigStart = html.indexOf("<div class=\"signature\">");
        if (sigStart >= 0) {
            int sigEnd = html.indexOf("</div>", sigStart);
            if (sigEnd >= 0) {
                System.out.println("Signature section HTML:");
                System.out.println(html.substring(sigStart, sigEnd + 6));
            }
        }
        return html;
    }

    private void show() {
        shell.open();
    }

    /**
     * Load an HTML resource from the classpath.
     */
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

    /**
     * Extract the title from HTML content.
     */
    private String extractTitle(String html) {
        int start = html.indexOf("<title>");
        int end = html.indexOf("</title>");
        if (start >= 0 && end > start) {
            return html.substring(start + 7, end);
        }
        // Try h1
        start = html.indexOf("<h1>");
        end = html.indexOf("</h1>");
        if (start >= 0 && end > start) {
            return html.substring(start + 4, end).replaceAll("<[^>]+>", "");
        }
        return null;
    }

    /**
     * Generate HTML for a "not found" page.
     */
    private String getNotFoundHtml(String docPath) {
        return "<!DOCTYPE html><html><head><title>Help Not Found</title>" +
               "<style>body { font-family: sans-serif; margin: 40px; text-align: center; }" +
               "h1 { color: #e74c3c; }</style></head>" +
               "<body><h1>Help Not Available</h1>" +
               "<p>No documentation found for: <code>" + docPath + "</code></p>" +
               "<p>This help page has not been created yet.</p></body></html>";
    }

    /**
     * Open a URL in the system default browser.
     * Tries multiple methods for cross-platform compatibility.
     */
    private static void openInSystemBrowser(String url) {
        System.out.println("Opening external URL: " + url);

        // Try Desktop API first (most reliable on macOS)
        if (Desktop.isDesktopSupported()) {
            Desktop desktop = Desktop.getDesktop();
            if (desktop.isSupported(Desktop.Action.BROWSE)) {
                try {
                    desktop.browse(new URI(url));
                    System.out.println("Opened via Desktop.browse");
                    return;
                } catch (Exception e) {
                    System.err.println("Desktop.browse failed: " + e.getMessage());
                }
            }
        }

        // Fall back to SWT Program.launch
        if (Program.launch(url)) {
            System.out.println("Opened via Program.launch");
            return;
        }

        // macOS-specific fallback: use 'open' command
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("mac")) {
            try {
                Runtime.getRuntime().exec(new String[]{"open", url});
                System.out.println("Opened via 'open' command");
                return;
            } catch (Exception e) {
                System.err.println("'open' command failed: " + e.getMessage());
            }
        }

        System.err.println("Failed to open URL in system browser: " + url);
    }
}
