package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Warp Affine (geometric transformation) node.
 */
@NodeInfo(name = "WarpAffine", category = "Transform", aliases = {"Warp Affine"})
public class WarpAffineNode extends ProcessingNode {
    private int translateX = 0;
    private int translateY = 0;
    private double rotation = 0.0;
    private double scale = 1.0;
    private int borderModeIndex = 0; // 0=constant, 1=replicate, 2=reflect, 3=wrap

    private static final String[] BORDER_MODE_NAMES = {
        "Constant (black)", "Replicate edge", "Reflect", "Wrap"
    };
    private static final int[] BORDER_MODES = {
        Core.BORDER_CONSTANT, Core.BORDER_REPLICATE, Core.BORDER_REFLECT, Core.BORDER_WRAP
    };

    public WarpAffineNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Warp Affine", x, y);
    }

    // Getters/setters for serialization
    public int getTranslateX() { return translateX; }
    public void setTranslateX(int v) { translateX = v; }
    public int getTranslateY() { return translateY; }
    public void setTranslateY(int v) { translateY = v; }
    public double getRotation() { return rotation; }
    public void setRotation(double v) { rotation = v; }
    public double getScale() { return scale; }
    public void setScale(double v) { scale = v; }
    public int getBorderModeIndex() { return borderModeIndex; }
    public void setBorderModeIndex(int v) { borderModeIndex = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        int height = input.rows();
        int width = input.cols();

        // Determine center point (always use image center)
        double cx = width / 2.0;
        double cy = height / 2.0;

        // Build transformation matrix
        // getRotationMatrix2D handles rotation and scale around center
        // Negate angle so positive = clockwise (more intuitive)
        Mat M = Imgproc.getRotationMatrix2D(new org.opencv.core.Point(cx, cy), -rotation, scale);

        // Add translation
        double[] row0 = M.get(0, 2);
        double[] row1 = M.get(1, 2);
        M.put(0, 2, row0[0] + translateX);
        M.put(1, 2, row1[0] + translateY);

        // Get border mode
        int borderMode = BORDER_MODES[borderModeIndex < BORDER_MODES.length ? borderModeIndex : 0];

        // Apply transformation
        Mat output = new Mat();
        Imgproc.warpAffine(input, output, M, new Size(width, height), Imgproc.INTER_LINEAR, borderMode);

        return output;
    }

    @Override
    public String getDescription() {
        return "Warp Affine: Translation, rotation, scaling\ncv2.warpAffine(src, M, dsize)";
    }

    @Override
    public String getDisplayName() {
        return "Warp Affine";
    }

    @Override
    public String getCategory() {
        return "Transform";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Warp Affine Properties");
        dialog.setLayout(new GridLayout(3, false));

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

        // Translate X
        new Label(dialog, SWT.NONE).setText("Translate X:");
        Scale txScale = new Scale(dialog, SWT.HORIZONTAL);
        txScale.setMinimum(0);
        txScale.setMaximum(1000);
        // Clamp slider position to valid range, but keep actual value
        int txSliderPos = Math.min(Math.max(translateX + 500, 0), 1000); // Offset to handle negative values
        txScale.setSelection(txSliderPos);
        txScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label txLabel = new Label(dialog, SWT.NONE);
        txLabel.setText(String.valueOf(translateX)); // Show real value
        txScale.addListener(SWT.Selection, e -> txLabel.setText(String.valueOf(txScale.getSelection() - 500)));

        // Translate Y
        new Label(dialog, SWT.NONE).setText("Translate Y:");
        Scale tyScale = new Scale(dialog, SWT.HORIZONTAL);
        tyScale.setMinimum(0);
        tyScale.setMaximum(1000);
        // Clamp slider position to valid range, but keep actual value
        int tySliderPos = Math.min(Math.max(translateY + 500, 0), 1000); // Offset to handle negative values
        tyScale.setSelection(tySliderPos);
        tyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label tyLabel = new Label(dialog, SWT.NONE);
        tyLabel.setText(String.valueOf(translateY)); // Show real value
        tyScale.addListener(SWT.Selection, e -> tyLabel.setText(String.valueOf(tyScale.getSelection() - 500)));

        // Rotation (-180 to 180)
        new Label(dialog, SWT.NONE).setText("Rotation:");
        Scale rotScale = new Scale(dialog, SWT.HORIZONTAL);
        rotScale.setMinimum(0);
        rotScale.setMaximum(360);
        // Clamp slider position to valid range, but keep actual value
        int rotSliderPos = Math.min(Math.max((int)rotation + 180, 0), 360); // Offset to handle negative values
        rotScale.setSelection(rotSliderPos);
        rotScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label rotLabel = new Label(dialog, SWT.NONE);
        rotLabel.setText(String.format("%.0f°", rotation)); // Show real value
        rotScale.addListener(SWT.Selection, e -> rotLabel.setText(String.format("%d°", rotScale.getSelection() - 180)));

        // Scale (0.1 to 4.0, scaled by 10)
        new Label(dialog, SWT.NONE).setText("Scale:");
        Scale scaleScale = new Scale(dialog, SWT.HORIZONTAL);
        scaleScale.setMinimum(1);
        scaleScale.setMaximum(40);
        // Clamp slider position to valid range, but keep actual value
        int scaleSliderPos = Math.min(Math.max((int)(scale * 10), 1), 40);
        scaleScale.setSelection(scaleSliderPos);
        scaleScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label scaleLabel = new Label(dialog, SWT.NONE);
        scaleLabel.setText(String.format("%.1fx", scale)); // Show real value
        scaleScale.addListener(SWT.Selection, e -> {
            double val = scaleScale.getSelection() / 10.0;
            scaleLabel.setText(String.format("%.1fx", val));
        });

        // Border Mode
        new Label(dialog, SWT.NONE).setText("Border Mode:");
        Combo borderCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        borderCombo.setItems(BORDER_MODE_NAMES);
        borderCombo.select(borderModeIndex < BORDER_MODE_NAMES.length ? borderModeIndex : 0);
        GridData borderGd = new GridData();
        borderGd.horizontalSpan = 2;
        borderCombo.setLayoutData(borderGd);

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
            translateX = txScale.getSelection() - 500;
            translateY = tyScale.getSelection() - 500;
            rotation = rotScale.getSelection() - 180;
            scale = scaleScale.getSelection() / 10.0;
            borderModeIndex = borderCombo.getSelectionIndex();
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
        json.addProperty("translateX", translateX);
        json.addProperty("translateY", translateY);
        json.addProperty("rotation", rotation);
        json.addProperty("scale", scale);
        json.addProperty("borderModeIndex", borderModeIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("translateX")) translateX = json.get("translateX").getAsInt();
        if (json.has("translateY")) translateY = json.get("translateY").getAsInt();
        if (json.has("rotation")) rotation = json.get("rotation").getAsDouble();
        if (json.has("scale")) scale = json.get("scale").getAsDouble();
        if (json.has("borderModeIndex")) borderModeIndex = json.get("borderModeIndex").getAsInt();
    }
}
