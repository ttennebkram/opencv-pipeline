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
 * Ellipse node - draws an ellipse on the image.
 */
@NodeInfo(name = "Ellipse", category = "Content", aliases = {})
public class EllipseNode extends ProcessingNode {
    private int centerX = 100, centerY = 100;
    private int axisX = 100, axisY = 50; // Half-axes
    private int angle = 0; // Rotation angle in degrees
    private int startAngle = 0, endAngle = 360; // Arc angles
    private int colorR = 0, colorG = 255, colorB = 0; // Green default
    private int thickness = 2;
    private boolean filled = false;

    public EllipseNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Ellipse", x, y);
    }

    // Getters/setters for serialization
    public int getCenterX() { return centerX; }
    public void setCenterX(int v) { centerX = v; }
    public int getCenterY() { return centerY; }
    public void setCenterY(int v) { centerY = v; }
    public int getAxisX() { return axisX; }
    public void setAxisX(int v) { axisX = v; }
    public int getAxisY() { return axisY; }
    public void setAxisY(int v) { axisY = v; }
    public int getAngle() { return angle; }
    public void setAngle(int v) { angle = v; }
    public int getStartAngle() { return startAngle; }
    public void setStartAngle(int v) { startAngle = v; }
    public int getEndAngle() { return endAngle; }
    public void setEndAngle(int v) { endAngle = v; }
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

        Imgproc.ellipse(output, new Point(absCenterX, absCenterY),
            new Size(axisX, axisY), angle, startAngle, endAngle, color, thick);

        return output;
    }

    @Override
    public String getDescription() {
        return "Draw Ellipse\ncv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)";
    }

    @Override
    public String getDisplayName() {
        return "Ellipse";
    }

    @Override
    public String getCategory() {
        return "Content";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Ellipse Properties");
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

        // Axis X (half-width)
        new Label(dialog, SWT.NONE).setText("Axis X:");
        Spinner axisXSpinner = new Spinner(dialog, SWT.BORDER);
        axisXSpinner.setMinimum(1);
        axisXSpinner.setMaximum(4096);
        axisXSpinner.setSelection(axisX);
        axisXSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Axis Y (half-height)
        new Label(dialog, SWT.NONE).setText("Axis Y:");
        Spinner axisYSpinner = new Spinner(dialog, SWT.BORDER);
        axisYSpinner.setMinimum(1);
        axisYSpinner.setMaximum(4096);
        axisYSpinner.setSelection(axisY);
        axisYSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Rotation Angle
        new Label(dialog, SWT.NONE).setText("Angle:");
        Spinner angleSpinner = new Spinner(dialog, SWT.BORDER);
        angleSpinner.setMinimum(0);
        angleSpinner.setMaximum(360);
        angleSpinner.setSelection(angle);
        angleSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Start Angle
        new Label(dialog, SWT.NONE).setText("Start Angle:");
        Spinner startAngleSpinner = new Spinner(dialog, SWT.BORDER);
        startAngleSpinner.setMinimum(0);
        startAngleSpinner.setMaximum(360);
        startAngleSpinner.setSelection(startAngle);
        startAngleSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // End Angle
        new Label(dialog, SWT.NONE).setText("End Angle:");
        Spinner endAngleSpinner = new Spinner(dialog, SWT.BORDER);
        endAngleSpinner.setMinimum(0);
        endAngleSpinner.setMaximum(360);
        endAngleSpinner.setSelection(endAngle);
        endAngleSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

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
        okBtn.addListener(SWT.Selection, e -> {
            centerX = centerXSpinner.getSelection();
            centerY = centerYSpinner.getSelection();
            axisX = axisXSpinner.getSelection();
            axisY = axisYSpinner.getSelection();
            angle = angleSpinner.getSelection();
            startAngle = startAngleSpinner.getSelection();
            endAngle = endAngleSpinner.getSelection();
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
        json.addProperty("axisX", axisX);
        json.addProperty("axisY", axisY);
        json.addProperty("angle", angle);
        json.addProperty("startAngle", startAngle);
        json.addProperty("endAngle", endAngle);
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
        if (json.has("axisX")) axisX = json.get("axisX").getAsInt();
        if (json.has("axisY")) axisY = json.get("axisY").getAsInt();
        if (json.has("angle")) angle = json.get("angle").getAsInt();
        if (json.has("startAngle")) startAngle = json.get("startAngle").getAsInt();
        if (json.has("endAngle")) endAngle = json.get("endAngle").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
        if (json.has("thickness")) thickness = json.get("thickness").getAsInt();
        if (json.has("filled")) filled = json.get("filled").getAsBoolean();
    }
}
