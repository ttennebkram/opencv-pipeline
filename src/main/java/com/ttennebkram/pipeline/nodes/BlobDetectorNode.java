package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.features2d.SimpleBlobDetector_Params;
import org.opencv.imgproc.Imgproc;

/**
 * Enhanced Blob Detection node with full SimpleBlobDetector parameters.
 */
@NodeInfo(
    name = "BlobDetector",
    category = "Detection",
    aliases = {"Blob Detector"}
)
public class BlobDetectorNode extends ProcessingNode {
    private int minThreshold = 10;
    private int maxThreshold = 200;
    private boolean showOriginal = true;

    // Filter by Area
    private boolean filterByArea = true;
    private int minArea = 100;
    private int maxArea = 5000;

    // Filter by Circularity
    private boolean filterByCircularity = false;
    private int minCircularity = 10; // 0.1 * 100

    // Filter by Convexity
    private boolean filterByConvexity = false;
    private int minConvexity = 87; // 0.87 * 100

    // Filter by Inertia
    private boolean filterByInertia = false;
    private int minInertiaRatio = 1; // 0.01 * 100

    // Filter by Color
    private boolean filterByColor = false;
    private int blobColor = 0; // 0 = dark, 255 = light

    // Drawing color
    private int colorR = 255, colorG = 0, colorB = 0;

    public BlobDetectorNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Blob Detector", x, y);
    }

    // Getters/setters for serialization
    public int getMinThreshold() { return minThreshold; }
    public void setMinThreshold(int v) { minThreshold = v; }
    public int getMaxThreshold() { return maxThreshold; }
    public void setMaxThreshold(int v) { maxThreshold = v; }
    public boolean getShowOriginal() { return showOriginal; }
    public void setShowOriginal(boolean v) { showOriginal = v; }

    public boolean isFilterByArea() { return filterByArea; }
    public void setFilterByArea(boolean v) { filterByArea = v; }
    public int getMinArea() { return minArea; }
    public void setMinArea(int v) { minArea = v; }
    public int getMaxArea() { return maxArea; }
    public void setMaxArea(int v) { maxArea = v; }

    public boolean isFilterByCircularity() { return filterByCircularity; }
    public void setFilterByCircularity(boolean v) { filterByCircularity = v; }
    public int getMinCircularity() { return minCircularity; }
    public void setMinCircularity(int v) { minCircularity = v; }

    public boolean isFilterByConvexity() { return filterByConvexity; }
    public void setFilterByConvexity(boolean v) { filterByConvexity = v; }
    public int getMinConvexity() { return minConvexity; }
    public void setMinConvexity(int v) { minConvexity = v; }

    public boolean isFilterByInertia() { return filterByInertia; }
    public void setFilterByInertia(boolean v) { filterByInertia = v; }
    public int getMinInertiaRatio() { return minInertiaRatio; }
    public void setMinInertiaRatio(int v) { minInertiaRatio = v; }

    public boolean isFilterByColor() { return filterByColor; }
    public void setFilterByColor(boolean v) { filterByColor = v; }
    public int getBlobColor() { return blobColor; }
    public void setBlobColor(int v) { blobColor = v; }

    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }

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

        // Create output image
        Mat result;
        if (showOriginal) {
            if (input.channels() == 1) {
                result = new Mat();
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }
        } else {
            result = Mat.zeros(input.size(), CvType.CV_8UC3);
        }

        // Set up SimpleBlobDetector parameters
        SimpleBlobDetector_Params params = new SimpleBlobDetector_Params();

        // Threshold parameters
        params.set_minThreshold(minThreshold);
        params.set_maxThreshold(maxThreshold);

        // Filter by Area
        params.set_filterByArea(filterByArea);
        params.set_minArea(minArea);
        params.set_maxArea(maxArea);

        // Filter by Circularity
        params.set_filterByCircularity(filterByCircularity);
        params.set_minCircularity((float)(minCircularity / 100.0));

        // Filter by Convexity
        params.set_filterByConvexity(filterByConvexity);
        params.set_minConvexity((float)(minConvexity / 100.0));

        // Filter by Inertia
        params.set_filterByInertia(filterByInertia);
        params.set_minInertiaRatio((float)(minInertiaRatio / 100.0));

        // Filter by Color - Note: blobColor parameter not available in Java bindings
        params.set_filterByColor(filterByColor);
        // params.set_blobColor(blobColor); // Not available in Java OpenCV bindings

        // Create detector and detect blobs
        SimpleBlobDetector detector = SimpleBlobDetector.create(params);
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        detector.detect(gray, keypoints);

        // Draw keypoints
        Scalar color = new Scalar(colorB, colorG, colorR);
        Features2d.drawKeypoints(result, keypoints, result, color,
            Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);

        // Cleanup
        gray.release();

        return result;
    }

    @Override
    public String getDescription() {
        return "Blob Detection\ncv2.SimpleBlobDetector_create(params)";
    }

    @Override
    public String getDisplayName() {
        return "Blob Detector";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Blob Detector Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Node name field
        Text nameText = addNameField(dialog, 3);

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep1 = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 3;
        sep1.setLayoutData(sepGd);

        // Show Original checkbox
        Button showOrigBtn = new Button(dialog, SWT.CHECK);
        showOrigBtn.setText("Show Original Background");
        showOrigBtn.setSelection(showOriginal);
        GridData showGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        showGd.horizontalSpan = 3;
        showOrigBtn.setLayoutData(showGd);

        // Min Threshold
        new Label(dialog, SWT.NONE).setText("Min Threshold:");
        Scale minThreshScale = new Scale(dialog, SWT.HORIZONTAL);
        minThreshScale.setMinimum(0);
        minThreshScale.setMaximum(255);
        minThreshScale.setSelection(minThreshold);
        minThreshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label minThreshLabel = new Label(dialog, SWT.NONE);
        minThreshLabel.setText(String.valueOf(minThreshold));
        minThreshScale.addListener(SWT.Selection, e -> minThreshLabel.setText(String.valueOf(minThreshScale.getSelection())));

        // Max Threshold
        new Label(dialog, SWT.NONE).setText("Max Threshold:");
        Scale maxThreshScale = new Scale(dialog, SWT.HORIZONTAL);
        maxThreshScale.setMinimum(0);
        maxThreshScale.setMaximum(255);
        maxThreshScale.setSelection(maxThreshold);
        maxThreshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label maxThreshLabel = new Label(dialog, SWT.NONE);
        maxThreshLabel.setText(String.valueOf(maxThreshold));
        maxThreshScale.addListener(SWT.Selection, e -> maxThreshLabel.setText(String.valueOf(maxThreshScale.getSelection())));

        // Filter by Area checkbox
        Button areaCheck = new Button(dialog, SWT.CHECK);
        areaCheck.setText("Filter by Area");
        areaCheck.setSelection(filterByArea);
        GridData areaGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        areaGd.horizontalSpan = 3;
        areaCheck.setLayoutData(areaGd);

        // Min Area
        new Label(dialog, SWT.NONE).setText("Min Area:");
        Scale minAreaScale = new Scale(dialog, SWT.HORIZONTAL);
        minAreaScale.setMinimum(1);
        minAreaScale.setMaximum(10000);
        minAreaScale.setSelection(minArea);
        minAreaScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label minAreaLabel = new Label(dialog, SWT.NONE);
        minAreaLabel.setText(String.valueOf(minArea));
        minAreaScale.addListener(SWT.Selection, e -> minAreaLabel.setText(String.valueOf(minAreaScale.getSelection())));

        // Max Area
        new Label(dialog, SWT.NONE).setText("Max Area:");
        Scale maxAreaScale = new Scale(dialog, SWT.HORIZONTAL);
        maxAreaScale.setMinimum(1);
        maxAreaScale.setMaximum(50000);
        maxAreaScale.setSelection(maxArea);
        maxAreaScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label maxAreaLabel = new Label(dialog, SWT.NONE);
        maxAreaLabel.setText(String.valueOf(maxArea));
        maxAreaScale.addListener(SWT.Selection, e -> maxAreaLabel.setText(String.valueOf(maxAreaScale.getSelection())));

        // Filter by Circularity checkbox
        Button circCheck = new Button(dialog, SWT.CHECK);
        circCheck.setText("Filter by Circularity");
        circCheck.setSelection(filterByCircularity);
        GridData circGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        circGd.horizontalSpan = 3;
        circCheck.setLayoutData(circGd);

        // Min Circularity
        new Label(dialog, SWT.NONE).setText("Min Circularity %:");
        Scale circScale = new Scale(dialog, SWT.HORIZONTAL);
        circScale.setMinimum(1);
        circScale.setMaximum(100);
        circScale.setSelection(minCircularity);
        circScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label circLabel = new Label(dialog, SWT.NONE);
        circLabel.setText(String.valueOf(minCircularity));
        circScale.addListener(SWT.Selection, e -> circLabel.setText(String.valueOf(circScale.getSelection())));

        // Filter by Convexity checkbox
        Button convCheck = new Button(dialog, SWT.CHECK);
        convCheck.setText("Filter by Convexity");
        convCheck.setSelection(filterByConvexity);
        GridData convGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        convGd.horizontalSpan = 3;
        convCheck.setLayoutData(convGd);

        // Min Convexity
        new Label(dialog, SWT.NONE).setText("Min Convexity %:");
        Scale convScale = new Scale(dialog, SWT.HORIZONTAL);
        convScale.setMinimum(1);
        convScale.setMaximum(100);
        convScale.setSelection(minConvexity);
        convScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label convLabel = new Label(dialog, SWT.NONE);
        convLabel.setText(String.valueOf(minConvexity));
        convScale.addListener(SWT.Selection, e -> convLabel.setText(String.valueOf(convScale.getSelection())));

        // Filter by Inertia checkbox
        Button inertiaCheck = new Button(dialog, SWT.CHECK);
        inertiaCheck.setText("Filter by Inertia");
        inertiaCheck.setSelection(filterByInertia);
        GridData inertiaGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        inertiaGd.horizontalSpan = 3;
        inertiaCheck.setLayoutData(inertiaGd);

        // Min Inertia Ratio
        new Label(dialog, SWT.NONE).setText("Min Inertia %:");
        Scale inertiaScale = new Scale(dialog, SWT.HORIZONTAL);
        inertiaScale.setMinimum(1);
        inertiaScale.setMaximum(100);
        inertiaScale.setSelection(minInertiaRatio);
        inertiaScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label inertiaLabel = new Label(dialog, SWT.NONE);
        inertiaLabel.setText(String.valueOf(minInertiaRatio));
        inertiaScale.addListener(SWT.Selection, e -> inertiaLabel.setText(String.valueOf(inertiaScale.getSelection())));

        // Filter by Color checkbox
        Button colorCheck = new Button(dialog, SWT.CHECK);
        colorCheck.setText("Filter by Color");
        colorCheck.setSelection(filterByColor);
        GridData colorFilterGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        colorFilterGd.horizontalSpan = 3;
        colorCheck.setLayoutData(colorFilterGd);

        // Blob Color (dark=0 or light=255)
        new Label(dialog, SWT.NONE).setText("Blob Color:");
        Combo colorCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        colorCombo.setItems(new String[] {"Dark (0)", "Light (255)"});
        colorCombo.select(blobColor == 0 ? 0 : 1);
        GridData colorComboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        colorComboGd.horizontalSpan = 2;
        colorCombo.setLayoutData(colorComboGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            saveNameField(nameText);
            showOriginal = showOrigBtn.getSelection();
            minThreshold = minThreshScale.getSelection();
            maxThreshold = maxThreshScale.getSelection();
            filterByArea = areaCheck.getSelection();
            minArea = minAreaScale.getSelection();
            maxArea = maxAreaScale.getSelection();
            filterByCircularity = circCheck.getSelection();
            minCircularity = circScale.getSelection();
            filterByConvexity = convCheck.getSelection();
            minConvexity = convScale.getSelection();
            filterByInertia = inertiaCheck.getSelection();
            minInertiaRatio = inertiaScale.getSelection();
            filterByColor = colorCheck.getSelection();
            blobColor = colorCombo.getSelectionIndex() == 0 ? 0 : 255;
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

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("minThreshold", minThreshold);
        json.addProperty("maxThreshold", maxThreshold);
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("filterByArea", filterByArea);
        json.addProperty("minArea", minArea);
        json.addProperty("maxArea", maxArea);
        json.addProperty("filterByCircularity", filterByCircularity);
        json.addProperty("minCircularity", minCircularity);
        json.addProperty("filterByConvexity", filterByConvexity);
        json.addProperty("minConvexity", minConvexity);
        json.addProperty("filterByInertia", filterByInertia);
        json.addProperty("minInertiaRatio", minInertiaRatio);
        json.addProperty("filterByColor", filterByColor);
        json.addProperty("blobColor", blobColor);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("minThreshold")) minThreshold = json.get("minThreshold").getAsInt();
        if (json.has("maxThreshold")) maxThreshold = json.get("maxThreshold").getAsInt();
        if (json.has("showOriginal")) showOriginal = json.get("showOriginal").getAsBoolean();
        if (json.has("filterByArea")) filterByArea = json.get("filterByArea").getAsBoolean();
        if (json.has("minArea")) minArea = json.get("minArea").getAsInt();
        if (json.has("maxArea")) maxArea = json.get("maxArea").getAsInt();
        if (json.has("filterByCircularity")) filterByCircularity = json.get("filterByCircularity").getAsBoolean();
        if (json.has("minCircularity")) minCircularity = json.get("minCircularity").getAsInt();
        if (json.has("filterByConvexity")) filterByConvexity = json.get("filterByConvexity").getAsBoolean();
        if (json.has("minConvexity")) minConvexity = json.get("minConvexity").getAsInt();
        if (json.has("filterByInertia")) filterByInertia = json.get("filterByInertia").getAsBoolean();
        if (json.has("minInertiaRatio")) minInertiaRatio = json.get("minInertiaRatio").getAsInt();
        if (json.has("filterByColor")) filterByColor = json.get("filterByColor").getAsBoolean();
        if (json.has("blobColor")) blobColor = json.get("blobColor").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
    }
}
