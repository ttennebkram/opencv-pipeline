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
 * Canny Edge Detection node.
 */
@NodeInfo(
    name = "CannyEdge",
    category = "Edges",
    aliases = {"Canny Edge", "Canny Edges"}
)
public class CannyEdgeNode extends ProcessingNode {
    private static final String[] APERTURE_SIZES = {"3", "5", "7"};
    private static final int[] APERTURE_VALUES = {3, 5, 7};

    private int threshold1 = 30;      // Lower threshold
    private int threshold2 = 150;     // Upper threshold
    private int apertureIndex = 0;    // Index into APERTURE_VALUES (default 3)
    private boolean l2Gradient = false;

    public CannyEdgeNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Canny Edges", x, y);
    }

    // Getters/setters for serialization
    public int getThreshold1() { return threshold1; }
    public void setThreshold1(int v) { threshold1 = v; }
    public int getThreshold2() { return threshold2; }
    public void setThreshold2(int v) { threshold2 = v; }
    public int getApertureIndex() { return apertureIndex; }
    public void setApertureIndex(int v) { apertureIndex = v; }
    public boolean isL2Gradient() { return l2Gradient; }
    public void setL2Gradient(boolean v) { l2Gradient = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Convert to grayscale if needed
        Mat gray;
        if (input.channels() == 3) {
            gray = new Mat();
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input;
        }

        // Apply Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, threshold1, threshold2,
                     APERTURE_VALUES[apertureIndex], l2Gradient);

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(edges, output, Imgproc.COLOR_GRAY2BGR);

        // Clean up temp mat if we created it
        if (gray != input) {
            gray.release();
        }
        edges.release();

        return output;
    }

    @Override
    public String getDescription() {
        return "Canny Edge Detection\ncv2.Canny(image, threshold1, threshold2)";
    }

    @Override
    public String getDisplayName() {
        return "Canny Edges";
    }

    @Override
    public String getCategory() {
        return "Edges";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Canny Edge Detection Properties");
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

        // Threshold 1 (lower)
        new Label(dialog, SWT.NONE).setText("Threshold 1 (lower):");
        Scale t1Scale = new Scale(dialog, SWT.HORIZONTAL);
        t1Scale.setMinimum(0);
        t1Scale.setMaximum(500);
        // Clamp slider position to valid range, but keep actual value
        int t1SliderPos = Math.min(Math.max(threshold1, 0), 500);
        t1Scale.setSelection(t1SliderPos);
        t1Scale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label t1Label = new Label(dialog, SWT.NONE);
        t1Label.setText(String.valueOf(threshold1)); // Show real value
        t1Scale.addListener(SWT.Selection, e -> t1Label.setText(String.valueOf(t1Scale.getSelection())));

        // Threshold 2 (upper)
        new Label(dialog, SWT.NONE).setText("Threshold 2 (upper):");
        Scale t2Scale = new Scale(dialog, SWT.HORIZONTAL);
        t2Scale.setMinimum(0);
        t2Scale.setMaximum(500);
        // Clamp slider position to valid range, but keep actual value
        int t2SliderPos = Math.min(Math.max(threshold2, 0), 500);
        t2Scale.setSelection(t2SliderPos);
        t2Scale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label t2Label = new Label(dialog, SWT.NONE);
        t2Label.setText(String.valueOf(threshold2)); // Show real value
        t2Scale.addListener(SWT.Selection, e -> t2Label.setText(String.valueOf(t2Scale.getSelection())));

        // Aperture Size (dropdown)
        new Label(dialog, SWT.NONE).setText("Aperture Size:");
        Combo apertureCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        apertureCombo.setItems(APERTURE_SIZES);
        apertureCombo.select(apertureIndex);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        apertureCombo.setLayoutData(comboGd);

        // L2 Gradient (checkbox)
        Button l2Check = new Button(dialog, SWT.CHECK);
        l2Check.setText("L2 Gradient");
        l2Check.setSelection(l2Gradient);
        GridData checkGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        checkGd.horizontalSpan = 3;
        l2Check.setLayoutData(checkGd);

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
            threshold1 = t1Scale.getSelection();
            threshold2 = t2Scale.getSelection();
            apertureIndex = apertureCombo.getSelectionIndex();
            l2Gradient = l2Check.getSelection();
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
        json.addProperty("threshold1", threshold1);
        json.addProperty("threshold2", threshold2);
        json.addProperty("apertureIndex", apertureIndex);
        json.addProperty("l2Gradient", l2Gradient);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("threshold1")) {
            threshold1 = json.get("threshold1").getAsInt();
        }
        if (json.has("threshold2")) {
            threshold2 = json.get("threshold2").getAsInt();
        }
        if (json.has("apertureIndex")) {
            apertureIndex = json.get("apertureIndex").getAsInt();
        }
        if (json.has("l2Gradient")) {
            l2Gradient = json.get("l2Gradient").getAsBoolean();
        }
    }
}
