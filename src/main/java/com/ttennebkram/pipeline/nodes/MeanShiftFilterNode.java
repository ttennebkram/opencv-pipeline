package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Mean Shift Filter node - color segmentation.
 */
@NodeInfo(
    name = "MeanShift",
    category = "Blur",
    aliases = {"Mean Shift", "Mean Shift Blur"}
)
public class MeanShiftFilterNode extends ProcessingNode {
    private int spatialRadius = 20;
    private int colorRadius = 40;
    private int maxLevel = 1;

    public MeanShiftFilterNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Mean Shift Blur", x, y);
    }

    // Getters/setters for serialization
    public int getSpatialRadius() { return spatialRadius; }
    public void setSpatialRadius(int v) { spatialRadius = v; }
    public int getColorRadius() { return colorRadius; }
    public void setColorRadius(int v) { colorRadius = v; }
    public int getMaxLevel() { return maxLevel; }
    public void setMaxLevel(int v) { maxLevel = v; }

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
        return "Mean Shift Filtering\ncv2.pyrMeanShiftFiltering(src, sp, sr)";
    }

    @Override
    public String getDisplayName() {
        return "Mean Shift Blur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Mean Shift Filter Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

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

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("spatialRadius", spatialRadius);
        json.addProperty("colorRadius", colorRadius);
        json.addProperty("maxLevel", maxLevel);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("spatialRadius")) spatialRadius = json.get("spatialRadius").getAsInt();
        if (json.has("colorRadius")) colorRadius = json.get("colorRadius").getAsInt();
        if (json.has("maxLevel")) maxLevel = json.get("maxLevel").getAsInt();
    }
}
