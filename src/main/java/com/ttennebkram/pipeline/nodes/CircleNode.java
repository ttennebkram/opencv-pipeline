package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Circle node - draws a circle on the image.
 */
@NodeInfo(name = "Circle", category = "Content", aliases = {})
public class CircleNode extends ProcessingNode {
    private int centerX = 100, centerY = 100;
    private int radius = 50;
    private int colorR = 0, colorG = 255, colorB = 0; // Green default
    private int thickness = 2;
    private boolean filled = false;

    public CircleNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Circle", x, y);
    }

    // Getters/setters for serialization
    public int getCenterX() { return centerX; }
    public void setCenterX(int v) { centerX = v; }
    public int getCenterY() { return centerY; }
    public void setCenterY(int v) { centerY = v; }
    public int getRadius() { return radius; }
    public void setRadius(int v) { radius = v; }
    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }
    public int getThickness() { return thickness; }
    public void setThickness(int v) { thickness = v; }
    public boolean isFilled() { return filled; }
    public void setFilled(boolean v) { filled = v; }

    // Helper to convert relative coords to absolute
    private int toAbsoluteX(int val, int imgWidth) {
        return val < 0 ? imgWidth + val : val;
    }

    private int toAbsoluteY(int val, int imgHeight) {
        return val < 0 ? imgHeight + val : val;
    }

    @Override
    public Mat process(Mat input) {
        Mat output = input.clone();
        int imgWidth = input.width();
        int imgHeight = input.height();

        Scalar color = new Scalar(colorB, colorG, colorR); // OpenCV uses BGR
        int thick = filled ? -1 : thickness;

        int absCenterX = toAbsoluteX(centerX, imgWidth);
        int absCenterY = toAbsoluteY(centerY, imgHeight);

        Imgproc.circle(output, new Point(absCenterX, absCenterY), radius, color, thick);

        return output;
    }

    @Override
    public String getDescription() {
        return "Draw Circle\ncv2.circle(img, center, radius, color, thickness)";
    }

    @Override
    public String getDisplayName() {
        return "Circle";
    }

    @Override
    public String getCategory() {
        return "Content";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Circle Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Center X
        new Label(dialog, SWT.NONE).setText("Center X:");
        Spinner centerXSpinner = new Spinner(dialog, SWT.BORDER);
        centerXSpinner.setMinimum(-4096);
        centerXSpinner.setMaximum(4096);
        centerXSpinner.setSelection(centerX);
        centerXSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Center Y
        new Label(dialog, SWT.NONE).setText("Center Y:");
        Spinner centerYSpinner = new Spinner(dialog, SWT.BORDER);
        centerYSpinner.setMinimum(-4096);
        centerYSpinner.setMaximum(4096);
        centerYSpinner.setSelection(centerY);
        centerYSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Radius
        new Label(dialog, SWT.NONE).setText("Radius:");
        Spinner radiusSpinner = new Spinner(dialog, SWT.BORDER);
        radiusSpinner.setMinimum(1);
        radiusSpinner.setMaximum(4096);
        radiusSpinner.setSelection(radius);
        radiusSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Color R
        new Label(dialog, SWT.NONE).setText("Red:");
        Spinner rSpinner = new Spinner(dialog, SWT.BORDER);
        rSpinner.setMinimum(0);
        rSpinner.setMaximum(255);
        rSpinner.setSelection(colorR);
        rSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Color G
        new Label(dialog, SWT.NONE).setText("Green:");
        Spinner gSpinner = new Spinner(dialog, SWT.BORDER);
        gSpinner.setMinimum(0);
        gSpinner.setMaximum(255);
        gSpinner.setSelection(colorG);
        gSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Color B
        new Label(dialog, SWT.NONE).setText("Blue:");
        Spinner bSpinner = new Spinner(dialog, SWT.BORDER);
        bSpinner.setMinimum(0);
        bSpinner.setMaximum(255);
        bSpinner.setSelection(colorB);
        bSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Thickness
        new Label(dialog, SWT.NONE).setText("Thickness:");
        Spinner thickSpinner = new Spinner(dialog, SWT.BORDER);
        thickSpinner.setMinimum(1);
        thickSpinner.setMaximum(50);
        thickSpinner.setSelection(thickness);
        thickSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Filled checkbox
        new Label(dialog, SWT.NONE).setText("Filled:");
        Button filledCheck = new Button(dialog, SWT.CHECK);
        filledCheck.setSelection(filled);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            centerX = centerXSpinner.getSelection();
            centerY = centerYSpinner.getSelection();
            radius = radiusSpinner.getSelection();
            colorR = rSpinner.getSelection();
            colorG = gSpinner.getSelection();
            colorB = bSpinner.getSelection();
            thickness = thickSpinner.getSelection();
            filled = filledCheck.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        org.eclipse.swt.graphics.Point cursorLoc = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursorLoc.x, cursorLoc.y);
        dialog.open();
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("centerX", centerX);
        json.addProperty("centerY", centerY);
        json.addProperty("radius", radius);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
        json.addProperty("thickness", thickness);
        json.addProperty("filled", filled);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("centerX")) centerX = json.get("centerX").getAsInt();
        if (json.has("centerY")) centerY = json.get("centerY").getAsInt();
        if (json.has("radius")) radius = json.get("radius").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
        if (json.has("thickness")) thickness = json.get("thickness").getAsInt();
        if (json.has("filled")) filled = json.get("filled").getAsBoolean();
    }
}
