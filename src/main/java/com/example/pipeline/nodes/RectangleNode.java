package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Rectangle node - draws a rectangle on the image.
 */
public class RectangleNode extends ProcessingNode {
    private int x1 = 50, y1 = 50;  // Top-left corner
    private int x2 = 200, y2 = 150; // Bottom-right corner
    private int colorR = 0, colorG = 255, colorB = 0; // Green default
    private int thickness = 2;
    private boolean filled = false;

    public RectangleNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Rectangle", x, y);
    }

    // Getters/setters for serialization
    public int getX1() { return x1; }
    public void setX1(int v) { x1 = v; }
    public int getY1() { return y1; }
    public void setY1(int v) { y1 = v; }
    public int getX2() { return x2; }
    public void setX2(int v) { x2 = v; }
    public int getY2() { return y2; }
    public void setY2(int v) { y2 = v; }
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

        int absX1 = toAbsoluteX(x1, imgWidth);
        int absY1 = toAbsoluteY(y1, imgHeight);
        int absX2 = toAbsoluteX(x2, imgWidth);
        int absY2 = toAbsoluteY(y2, imgHeight);

        Imgproc.rectangle(output, new Point(absX1, absY1), new Point(absX2, absY2), color, thick);

        return output;
    }

    @Override
    public String getDescription() {
        return "Draw Rectangle\ncv2.rectangle(img, pt1, pt2, color, thickness)";
    }

    @Override
    public String getDisplayName() {
        return "Rectangle";
    }

    @Override
    public String getCategory() {
        return "Content";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Rectangle Properties");
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

        // X1
        new Label(dialog, SWT.NONE).setText("X1:");
        Spinner x1Spinner = new Spinner(dialog, SWT.BORDER);
        x1Spinner.setMinimum(-4096);
        x1Spinner.setMaximum(4096);
        x1Spinner.setSelection(x1);
        x1Spinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Y1
        new Label(dialog, SWT.NONE).setText("Y1:");
        Spinner y1Spinner = new Spinner(dialog, SWT.BORDER);
        y1Spinner.setMinimum(-4096);
        y1Spinner.setMaximum(4096);
        y1Spinner.setSelection(y1);
        y1Spinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // X2
        new Label(dialog, SWT.NONE).setText("X2:");
        Spinner x2Spinner = new Spinner(dialog, SWT.BORDER);
        x2Spinner.setMinimum(-4096);
        x2Spinner.setMaximum(4096);
        x2Spinner.setSelection(x2);
        x2Spinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Y2
        new Label(dialog, SWT.NONE).setText("Y2:");
        Spinner y2Spinner = new Spinner(dialog, SWT.BORDER);
        y2Spinner.setMinimum(-4096);
        y2Spinner.setMaximum(4096);
        y2Spinner.setSelection(y2);
        y2Spinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

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
            x1 = x1Spinner.getSelection();
            y1 = y1Spinner.getSelection();
            x2 = x2Spinner.getSelection();
            y2 = y2Spinner.getSelection();
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
}
