package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import com.ttennebkram.pipeline.fx.FXWebcamSource;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Calibration processor for projector/display alignment.
 *
 * This node:
 * 1. Outputs a calibration pattern (white rectangle) to be displayed on the projector
 * 2. Captures camera input to see how the pattern appears in the real world
 * 3. Detects the projected rectangle corners in the camera image
 * 4. Computes homography matrices (forward and inverse) for warping
 * 5. Applies the forward warp to input images to pre-distort them for correct projection
 *
 * The node has an optional input - if connected, it warps that input for output.
 * If no input, it outputs the calibration pattern.
 *
 * Background mode:
 * - Black: calibration pattern on black background (best for corner detection)
 * - Input: calibration pattern overlaid on input image (for live preview during calibration)
 */
@FXProcessorInfo(
    nodeType = "Calibrate",
    displayName = "Calibrate",
    category = "Transform",
    description = "Projector calibration\nDetects corners & computes warp",
    isSource = false  // Has optional input
)
public class CalibrateProcessor extends FXCameraProcessorBase {

    // Calibration settings
    private int patternMargin = 100;  // Pixels from edge for calibration rectangle
    private boolean useInputBackground = false;  // false = black, true = input image
    private int outputWidth = 1920;  // Output resolution for projector
    private int outputHeight = 1080;

    // Computed homography matrices (3x3)
    private Mat forwardHomography = null;  // Camera space -> Projector space
    private Mat inverseHomography = null;  // Projector space -> Camera space

    // Calibration state
    private boolean calibrationLocked = false;
    private int cornersFound = 0;
    private double calibrationError = 0.0;
    private Point[] detectedCorners = null;  // The 4 corners found in camera image
    private Point[] expectedCorners = null;  // Where we expect the corners to be

    // Camera reference (managed externally like WebcamSource)
    private transient FXWebcamSource webcamSource = null;
    private transient Mat lastCameraFrame = null;

    // Status text for display on node
    private String statusText = "Not calibrated";

    private static final String[] OUTPUT_RESOLUTIONS = {"1920x1080", "1280x720", "2560x1440", "3840x2160"};

    // UI components
    private CameraSettingsUI cameraUI;

    public CalibrateProcessor() {
        // Default mirror to false for calibration (usually don't want mirror)
        mirrorHorizontal = false;
    }

    @Override
    public String getNodeType() {
        return "Calibrate";
    }

    @Override
    public String getCategory() {
        return "Transform";
    }

    @Override
    public String getDescription() {
        return "Calibrate\nProjector alignment via camera feedback";
    }

    @Override
    public Mat process(Mat input) {
        // Create output image at projector resolution
        Mat output = new Mat(outputHeight, outputWidth, CvType.CV_8UC3);

        // If we have a homography and input, apply the warp (real-time - always when available)
        if (forwardHomography != null && input != null && !input.empty()) {
            Imgproc.warpPerspective(input, output, forwardHomography, new Size(outputWidth, outputHeight));
        } else if (input != null && !input.empty() && useInputBackground) {
            // No calibration yet, just resize input as background
            Imgproc.resize(input, output, new Size(outputWidth, outputHeight));
        } else {
            // No input or calibration - show black background
            output.setTo(new Scalar(0, 0, 0));
        }

        // Always overlay calibration pattern so camera can track it
        overlayCalibrationRectangle(output);

        // Update status text for node display
        updateStatusText();

        return output;
    }

    /**
     * Draw calibration rectangle directly on output image.
     */
    private void overlayCalibrationRectangle(Mat image) {
        int x1 = patternMargin;
        int y1 = patternMargin;
        int x2 = outputWidth - patternMargin;
        int y2 = outputHeight - patternMargin;

        // Draw thick white rectangle border
        Imgproc.rectangle(image, new Point(x1, y1), new Point(x2, y2),
                          new Scalar(255, 255, 255), 5);

        // Store expected corners (clockwise from top-left)
        expectedCorners = new Point[] {
            new Point(x1, y1),  // Top-left
            new Point(x2, y1),  // Top-right
            new Point(x2, y2),  // Bottom-right
            new Point(x1, y2)   // Bottom-left
        };
    }

    /**
     * Generate the calibration pattern: white rectangle on black background.
     */
    private Mat generateCalibrationPattern() {
        Mat pattern = Mat.zeros(outputHeight, outputWidth, CvType.CV_8UC3);

        // Draw white rectangle with margin from edges
        int x1 = patternMargin;
        int y1 = patternMargin;
        int x2 = outputWidth - patternMargin;
        int y2 = outputHeight - patternMargin;

        // Draw thick white rectangle border
        Imgproc.rectangle(pattern, new Point(x1, y1), new Point(x2, y2),
                          new Scalar(255, 255, 255), 5);

        // Store expected corners (clockwise from top-left)
        expectedCorners = new Point[] {
            new Point(x1, y1),  // Top-left
            new Point(x2, y1),  // Top-right
            new Point(x2, y2),  // Bottom-right
            new Point(x1, y2)   // Bottom-left
        };

        return pattern;
    }

    /**
     * Overlay calibration pattern onto an image (just the rectangle border).
     */
    private void overlayPattern(Mat image, Mat pattern) {
        // Simple overlay - draw the rectangle on the image
        int x1 = patternMargin;
        int y1 = patternMargin;
        int x2 = outputWidth - patternMargin;
        int y2 = outputHeight - patternMargin;

        Imgproc.rectangle(image, new Point(x1, y1), new Point(x2, y2),
                          new Scalar(255, 255, 255), 5);
    }

    /**
     * Process camera frame to detect calibration rectangle corners.
     * Called from the pipeline executor when camera frames are available.
     */
    public void processCameraFrame(Mat cameraFrame) {
        if (cameraFrame == null || cameraFrame.empty()) {
            return;
        }

        // Store for preview
        if (lastCameraFrame != null) {
            lastCameraFrame.release();
        }
        lastCameraFrame = cameraFrame.clone();

        // Detect corners
        detectCorners(cameraFrame);

        // If we found 4 corners, compute homography
        if (cornersFound == 4 && detectedCorners != null && expectedCorners != null) {
            computeHomography();
        }
    }

    /**
     * Detect the calibration rectangle corners in the camera image.
     * Uses hierarchical contour detection to find nested rectangles
     * (the calibration rectangle inside the window frame).
     */
    private void detectCorners(Mat frame) {
        // Convert to grayscale
        Mat gray = new Mat();
        if (frame.channels() == 3) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            frame.copyTo(gray);
        }

        // Threshold to find bright regions (the white rectangle)
        Mat thresh = new Mat();
        Imgproc.threshold(gray, thresh, 200, 255, Imgproc.THRESH_BINARY);

        // Find contours with hierarchy to detect nested rectangles
        // RETR_TREE gives us parent-child relationships
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy,
                             Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        // Calculate expected aspect ratio of our calibration rectangle
        double expectedAspect = (double)(outputWidth - 2 * patternMargin) / (outputHeight - 2 * patternMargin);

        // Find all quadrilaterals and score them
        List<MatOfPoint2f> quads = new ArrayList<>();
        List<Double> scores = new ArrayList<>();

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            double area = Imgproc.contourArea(contour);
            if (area < 1000) continue;  // Skip small contours

            // Approximate to polygon
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            double epsilon = 0.02 * Imgproc.arcLength(contour2f, true);
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

            // Check if it's a quadrilateral
            if (approx.rows() == 4) {
                // Calculate aspect ratio of this quadrilateral
                Rect boundingRect = Imgproc.boundingRect(contour);
                double aspect = (double) boundingRect.width / boundingRect.height;

                // Score based on aspect ratio match (lower is better)
                double aspectScore = Math.abs(aspect - expectedAspect);

                // Check if this contour has a parent (is nested inside another)
                // hierarchy format: [next, prev, firstChild, parent]
                double[] h = hierarchy.get(0, i);
                boolean hasParent = h != null && h[3] >= 0;

                // Prefer nested rectangles (inside window frame) and good aspect ratio
                double score = aspectScore;
                if (hasParent) {
                    score -= 0.5;  // Bonus for being nested (likely inside window frame)
                }

                quads.add(approx);
                scores.add(score);
            } else {
                approx.release();
            }
            contour2f.release();
        }

        // Find the best quadrilateral (lowest score)
        MatOfPoint2f bestContour = null;
        double bestScore = Double.MAX_VALUE;
        for (int i = 0; i < quads.size(); i++) {
            if (scores.get(i) < bestScore) {
                if (bestContour != null) bestContour.release();
                bestContour = quads.get(i);
                bestScore = scores.get(i);
            } else {
                quads.get(i).release();
            }
        }

        // Extract corners from best contour
        if (bestContour != null) {
            Point[] points = bestContour.toArray();
            // Sort corners: top-left, top-right, bottom-right, bottom-left
            detectedCorners = sortCorners(points);
            cornersFound = 4;
            bestContour.release();
        } else {
            cornersFound = 0;
            detectedCorners = null;
        }

        // Cleanup
        gray.release();
        thresh.release();
        hierarchy.release();
        for (MatOfPoint c : contours) c.release();
    }

    /**
     * Sort 4 corner points into consistent order: TL, TR, BR, BL.
     */
    private Point[] sortCorners(Point[] points) {
        if (points.length != 4) return points;

        // Find centroid
        double cx = 0, cy = 0;
        for (Point p : points) {
            cx += p.x;
            cy += p.y;
        }
        cx /= 4;
        cy /= 4;

        // Classify points by quadrant relative to centroid
        Point tl = null, tr = null, br = null, bl = null;
        for (Point p : points) {
            if (p.x < cx && p.y < cy) tl = p;
            else if (p.x >= cx && p.y < cy) tr = p;
            else if (p.x >= cx && p.y >= cy) br = p;
            else bl = p;
        }

        // Handle edge cases where points don't fall neatly into quadrants
        if (tl == null || tr == null || br == null || bl == null) {
            return points;  // Return unsorted if classification fails
        }

        return new Point[] { tl, tr, br, bl };
    }

    /**
     * Compute homography matrices from detected corners.
     */
    private void computeHomography() {
        if (detectedCorners == null || expectedCorners == null) {
            return;
        }

        // Create point matrices for findHomography
        MatOfPoint2f srcPoints = new MatOfPoint2f(detectedCorners);
        MatOfPoint2f dstPoints = new MatOfPoint2f(expectedCorners);

        // Compute forward homography (camera -> projector)
        Mat h = Calib3d.findHomography(srcPoints, dstPoints);
        if (!h.empty()) {
            if (forwardHomography != null) forwardHomography.release();
            forwardHomography = h;

            // Compute inverse
            if (inverseHomography != null) inverseHomography.release();
            inverseHomography = forwardHomography.inv();

            // Calculate reprojection error
            calibrationError = calculateReprojectionError(srcPoints, dstPoints, forwardHomography);

            // Lock calibration if error is acceptable
            calibrationLocked = calibrationError < 10.0;  // Less than 10 pixels error
        }

        srcPoints.release();
        dstPoints.release();
    }

    /**
     * Calculate average reprojection error.
     */
    private double calculateReprojectionError(MatOfPoint2f src, MatOfPoint2f dst, Mat H) {
        MatOfPoint2f projected = new MatOfPoint2f();
        Core.perspectiveTransform(src, projected, H);

        Point[] projPoints = projected.toArray();
        Point[] dstPoints = dst.toArray();

        double totalError = 0;
        for (int i = 0; i < projPoints.length; i++) {
            double dx = projPoints[i].x - dstPoints[i].x;
            double dy = projPoints[i].y - dstPoints[i].y;
            totalError += Math.sqrt(dx * dx + dy * dy);
        }

        projected.release();
        return totalError / projPoints.length;
    }

    /**
     * Update status text for node display.
     */
    private void updateStatusText() {
        if (calibrationLocked) {
            statusText = String.format("Locked (%.1fpx)", calibrationError);
        } else if (cornersFound == 4) {
            statusText = String.format("Tracking %d/4 (%.1fpx)", cornersFound, calibrationError);
        } else {
            statusText = String.format("Detecting %d/4", cornersFound);
        }

        // Update fxNode if available
        if (fxNode != null) {
            fxNode.statusText = statusText;
        }
    }

    /**
     * Get the forward homography matrix as a 2D array for display.
     */
    public double[][] getForwardHomographyArray() {
        if (forwardHomography == null || forwardHomography.empty()) {
            return null;
        }
        double[][] result = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[i][j] = forwardHomography.get(i, j)[0];
            }
        }
        return result;
    }

    /**
     * Get the inverse homography matrix as a 2D array for display.
     */
    public double[][] getInverseHomographyArray() {
        if (inverseHomography == null || inverseHomography.empty()) {
            return null;
        }
        double[][] result = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[i][j] = inverseHomography.get(i, j)[0];
            }
        }
        return result;
    }

    // Getters for status display
    public boolean isCalibrationLocked() { return calibrationLocked; }
    public int getCornersFound() { return cornersFound; }
    public double getCalibrationError() { return calibrationError; }
    public String getStatusText() { return statusText; }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription("Projector Calibration\n\n" +
            "Displays a calibration pattern on the projector output.\n" +
            "Camera captures the projected pattern to compute alignment.");

        // Camera settings from base class
        cameraUI = addCameraSettingsUI(dialog, "Camera Settings");

        // Calibration settings
        dialog.addDescription("\nCalibration Settings");
        Spinner<Integer> marginSpinner = dialog.addSpinner("Pattern Margin (px):", 10, 500, patternMargin);

        // Output settings
        dialog.addDescription("\nOutput Settings");
        ComboBox<String> outResCombo = dialog.addComboBox("Output Resolution:", OUTPUT_RESOLUTIONS,
                getOutputResolutionString());
        CheckBox bgCheck = dialog.addCheckbox("Use Input as Background", useInputBackground);

        dialog.setOnOk(() -> {
            applyCameraSettingsFromUI(cameraUI);
            patternMargin = marginSpinner.getValue();
            useInputBackground = bgCheck.isSelected();
            parseOutputResolution(outResCombo.getValue());
        });
    }

    private String getOutputResolutionString() {
        return outputWidth + "x" + outputHeight;
    }

    private void parseOutputResolution(String res) {
        if (res == null) return;
        String[] parts = res.split("x");
        if (parts.length == 2) {
            try {
                outputWidth = Integer.parseInt(parts[0]);
                outputHeight = Integer.parseInt(parts[1]);
            } catch (NumberFormatException e) {
                // Keep current values
            }
        }
    }

    @Override
    public void serializeProperties(JsonObject json) {
        serializeCameraProperties(json);
        json.addProperty("patternMargin", patternMargin);
        json.addProperty("useInputBackground", useInputBackground);
        json.addProperty("outputWidth", outputWidth);
        json.addProperty("outputHeight", outputHeight);

        // Save homography matrices if computed
        if (forwardHomography != null && !forwardHomography.empty()) {
            JsonArray fwdArray = new JsonArray();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    fwdArray.add(forwardHomography.get(i, j)[0]);
                }
            }
            json.add("forwardHomography", fwdArray);
        }

        if (inverseHomography != null && !inverseHomography.empty()) {
            JsonArray invArray = new JsonArray();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    invArray.add(inverseHomography.get(i, j)[0]);
                }
            }
            json.add("inverseHomography", invArray);
        }
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        deserializeCameraProperties(json);
        patternMargin = getJsonInt(json, "patternMargin", 100);
        useInputBackground = json.has("useInputBackground") && json.get("useInputBackground").getAsBoolean();
        outputWidth = getJsonInt(json, "outputWidth", 1920);
        outputHeight = getJsonInt(json, "outputHeight", 1080);

        // Load homography matrices if present
        if (json.has("forwardHomography")) {
            JsonArray arr = json.getAsJsonArray("forwardHomography");
            if (arr.size() == 9) {
                forwardHomography = new Mat(3, 3, CvType.CV_64FC1);
                int idx = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        forwardHomography.put(i, j, arr.get(idx++).getAsDouble());
                    }
                }
            }
        }

        if (json.has("inverseHomography")) {
            JsonArray arr = json.getAsJsonArray("inverseHomography");
            if (arr.size() == 9) {
                inverseHomography = new Mat(3, 3, CvType.CV_64FC1);
                int idx = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        inverseHomography.put(i, j, arr.get(idx++).getAsDouble());
                    }
                }
                calibrationLocked = true;  // If we have saved matrices, consider it locked
            }
        }
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        syncCameraPropertiesFromFXNode(node);
        patternMargin = getInt(node.properties, "patternMargin", 100);
        useInputBackground = getBool(node.properties, "useInputBackground", false);
        outputWidth = getInt(node.properties, "outputWidth", 1920);
        outputHeight = getInt(node.properties, "outputHeight", 1080);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        syncCameraPropertiesToFXNode(node);
        node.properties.put("patternMargin", patternMargin);
        node.properties.put("useInputBackground", useInputBackground);
        node.properties.put("outputWidth", outputWidth);
        node.properties.put("outputHeight", outputHeight);
    }

    /**
     * Release OpenCV resources.
     */
    public void release() {
        if (forwardHomography != null) {
            forwardHomography.release();
            forwardHomography = null;
        }
        if (inverseHomography != null) {
            inverseHomography.release();
            inverseHomography = null;
        }
        if (lastCameraFrame != null) {
            lastCameraFrame.release();
            lastCameraFrame = null;
        }
    }
}
