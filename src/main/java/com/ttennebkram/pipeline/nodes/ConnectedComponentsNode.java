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
import org.opencv.imgproc.Imgproc;

import java.util.Random;

/**
 * Connected Components labeling node.
 */
@NodeInfo(name = "ConnectedComponents", category = "Detection", aliases = {"Connected Components"})
public class ConnectedComponentsNode extends ProcessingNode {
    private int threshold = 127;
    private boolean invertThreshold = false;
    private int connectivity = 8;
    private int minArea = 0;

    public ConnectedComponentsNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Connected Components", x, y);
    }

    // Getters/setters for serialization
    public int getThreshold() { return threshold; }
    public void setThreshold(int v) { threshold = v; }
    public boolean isInvertThreshold() { return invertThreshold; }
    public void setInvertThreshold(boolean v) { invertThreshold = v; }
    public int getConnectivity() { return connectivity; }
    public void setConnectivity(int v) { connectivity = v; }
    public int getMinArea() { return minArea; }
    public void setMinArea(int v) { minArea = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        Mat gray = null;
        Mat binary = null;
        Mat labels = null;
        Mat stats = null;
        Mat centroids = null;
        Mat result = null;

        try {
            // Convert to grayscale
            gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Apply threshold
            binary = new Mat();
            int threshType = invertThreshold ? Imgproc.THRESH_BINARY_INV : Imgproc.THRESH_BINARY;
            Imgproc.threshold(gray, binary, threshold, 255, threshType);

            // Get connected components with stats
            labels = new Mat();
            stats = new Mat();
            centroids = new Mat();
            int numLabels = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, connectivity, CvType.CV_32S);

            // Generate random colors for each label (with consistent seed)
            Random rand = new Random(42);
            int[][] colors = new int[numLabels][3];
            colors[0] = new int[]{0, 0, 0}; // Background is black
            for (int i = 1; i < numLabels; i++) {
                colors[i] = new int[]{rand.nextInt(256), rand.nextInt(256), rand.nextInt(256)};
            }

            // Filter by min area - set small components to black
            if (minArea > 0) {
                for (int i = 1; i < numLabels; i++) {
                    int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
                    if (area < minArea) {
                        colors[i] = new int[]{0, 0, 0};
                    }
                }
            }

            // Create colored output
            result = new Mat(input.rows(), input.cols(), CvType.CV_8UC3);
            for (int row = 0; row < labels.rows(); row++) {
                for (int col = 0; col < labels.cols(); col++) {
                    int label = (int) labels.get(row, col)[0];
                    result.put(row, col, colors[label][0], colors[label][1], colors[label][2]);
                }
            }

            return result;
        } finally {
            // Release intermediate Mats (but not result which is returned)
            if (gray != null) gray.release();
            if (binary != null) binary.release();
            if (labels != null) labels.release();
            if (stats != null) stats.release();
            if (centroids != null) centroids.release();
        }
    }

    @Override
    public String getDescription() {
        return "Connected Components: Label connected regions\ncv2.connectedComponentsWithStats(image)";
    }

    @Override
    public String getDisplayName() {
        return "Connected Components";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Connected Components Properties");
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
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 3;
        sep.setLayoutData(sepGd);

        // Threshold
        new Label(dialog, SWT.NONE).setText("Threshold:");
        Scale thresholdScale = new Scale(dialog, SWT.HORIZONTAL);
        thresholdScale.setMinimum(0);
        thresholdScale.setMaximum(255);
        // Clamp slider position to valid range, but keep actual value
        int thresholdSliderPos = Math.min(Math.max(threshold, 0), 255);
        thresholdScale.setSelection(thresholdSliderPos);
        thresholdScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label thresholdLabel = new Label(dialog, SWT.NONE);
        thresholdLabel.setText(String.valueOf(threshold)); // Show real value
        thresholdScale.addListener(SWT.Selection, e -> thresholdLabel.setText(String.valueOf(thresholdScale.getSelection())));

        // Invert Threshold
        new Label(dialog, SWT.NONE).setText("Invert Threshold:");
        Button invertCheck = new Button(dialog, SWT.CHECK);
        invertCheck.setSelection(invertThreshold);
        GridData invertGd = new GridData();
        invertGd.horizontalSpan = 2;
        invertCheck.setLayoutData(invertGd);

        // Connectivity
        new Label(dialog, SWT.NONE).setText("Connectivity:");
        Combo connectivityCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        connectivityCombo.setItems("4", "8");
        connectivityCombo.select(connectivity == 4 ? 0 : 1);
        GridData connGd = new GridData();
        connGd.horizontalSpan = 2;
        connectivityCombo.setLayoutData(connGd);

        // Min Area
        new Label(dialog, SWT.NONE).setText("Min Area:");
        Scale minAreaScale = new Scale(dialog, SWT.HORIZONTAL);
        minAreaScale.setMinimum(0);
        minAreaScale.setMaximum(1000);
        // Clamp slider position to valid range, but keep actual value
        int minAreaSliderPos = Math.min(Math.max(minArea, 0), 1000);
        minAreaScale.setSelection(minAreaSliderPos);
        minAreaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label minAreaLabel = new Label(dialog, SWT.NONE);
        minAreaLabel.setText(String.valueOf(minArea)); // Show real value
        minAreaScale.addListener(SWT.Selection, e -> minAreaLabel.setText(String.valueOf(minAreaScale.getSelection())));

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
            threshold = thresholdScale.getSelection();
            invertThreshold = invertCheck.getSelection();
            connectivity = connectivityCombo.getSelectionIndex() == 0 ? 4 : 8;
            minArea = minAreaScale.getSelection();
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

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("threshold", threshold);
        json.addProperty("invertThreshold", invertThreshold);
        json.addProperty("connectivity", connectivity);
        json.addProperty("minArea", minArea);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("threshold")) threshold = json.get("threshold").getAsInt();
        if (json.has("invertThreshold")) invertThreshold = json.get("invertThreshold").getAsBoolean();
        if (json.has("connectivity")) connectivity = json.get("connectivity").getAsInt();
        if (json.has("minArea")) minArea = json.get("minArea").getAsInt();
    }
}
