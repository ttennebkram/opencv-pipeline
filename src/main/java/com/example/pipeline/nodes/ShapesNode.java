package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Shapes node - draws geometric shapes on the image.
 */
public class ShapesNode extends ProcessingNode {
    private static final String[] SHAPE_NAMES = {
        "Rectangle", "Circle", "Ellipse", "Line", "Arrow"
    };

    private int shapeIndex = 0; // Default to Rectangle
    private int x1 = 50, y1 = 50;  // Start point
    private int x2 = 200, y2 = 150; // End point (or width/height for shapes)
    private int circleRadius = 50; // Radius for circle shape
    private int colorR = 0, colorG = 255, colorB = 0; // Green default
    private int thickness = 2;
    private boolean filled = false;

    public ShapesNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Shapes", x, y);
    }

    // Getters/setters for serialization
    public int getShapeIndex() { return shapeIndex; }
    public void setShapeIndex(int v) { shapeIndex = v; }
    public int getX1() { return x1; }
    public void setX1(int v) { x1 = v; }
    public int getY1() { return y1; }
    public void setY1(int v) { y1 = v; }
    public int getX2() { return x2; }
    public void setX2(int v) { x2 = v; }
    public int getY2() { return y2; }
    public void setY2(int v) { y2 = v; }
    public int getCircleRadius() { return circleRadius; }
    public void setCircleRadius(int v) { circleRadius = v; }
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

    // Convert coordinate to absolute position (negative values are relative to right/bottom edge)
    private int toAbsoluteX(int x, int width) {
        return x < 0 ? width + x + 1 : x;
    }

    private int toAbsoluteY(int y, int height) {
        return y < 0 ? height + y + 1 : y;
    }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        Mat output = input.clone();
        Scalar color = new Scalar(colorB, colorG, colorR); // BGR order
        int thick = filled ? -1 : thickness;

        // Convert coordinates (negative values are relative to right/bottom)
        int imgWidth = input.width();
        int imgHeight = input.height();
        int absX1 = toAbsoluteX(x1, imgWidth);
        int absY1 = toAbsoluteY(y1, imgHeight);
        int absX2 = toAbsoluteX(x2, imgWidth);
        int absY2 = toAbsoluteY(y2, imgHeight);

        switch (shapeIndex) {
            case 0: // Rectangle
                Imgproc.rectangle(output, new Point(absX1, absY1), new Point(absX2, absY2), color, thick);
                break;
            case 1: // Circle
                Imgproc.circle(output, new Point(absX1, absY1), circleRadius, color, thick);
                break;
            case 2: // Ellipse
                int axisX = Math.abs(absX2 - absX1) / 2;
                int axisY = Math.abs(absY2 - absY1) / 2;
                int centerX = (absX1 + absX2) / 2;
                int centerY = (absY1 + absY2) / 2;
                Imgproc.ellipse(output, new Point(centerX, centerY), new Size(axisX, axisY),
                    0, 0, 360, color, thick);
                break;
            case 3: // Line
                Imgproc.line(output, new Point(absX1, absY1), new Point(absX2, absY2), color, thickness);
                break;
            case 4: // Arrow
                Imgproc.arrowedLine(output, new Point(absX1, absY1), new Point(absX2, absY2), color, thickness);
                break;
        }

        return output;
    }

    @Override
    public String getDescription() {
        return "Draw Shapes\ncv2.rectangle(), circle(), ellipse(), line()";
    }

    @Override
    public String getDisplayName() {
        return "Shapes";
    }

    @Override
    public String getCategory() {
        return "Content";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Shapes Properties");
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

        // Shape type
        new Label(dialog, SWT.NONE).setText("Shape:");
        Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        shapeCombo.setItems(SHAPE_NAMES);
        shapeCombo.select(shapeIndex);
        shapeCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // X1 (negative values are relative to right edge)
        new Label(dialog, SWT.NONE).setText("X1:");
        Spinner x1Spinner = new Spinner(dialog, SWT.BORDER);
        x1Spinner.setMinimum(-4096);
        x1Spinner.setMaximum(4096);
        x1Spinner.setSelection(x1);
        x1Spinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Y1 (negative values are relative to bottom edge)
        new Label(dialog, SWT.NONE).setText("Y1:");
        Spinner y1Spinner = new Spinner(dialog, SWT.BORDER);
        y1Spinner.setMinimum(-4096);
        y1Spinner.setMaximum(4096);
        y1Spinner.setSelection(y1);
        y1Spinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // X2 (negative values are relative to right edge)
        Label x2Label = new Label(dialog, SWT.NONE);
        x2Label.setText("X2:");
        GridData x2LabelGd = new GridData();
        x2Label.setLayoutData(x2LabelGd);
        Spinner x2Spinner = new Spinner(dialog, SWT.BORDER);
        x2Spinner.setMinimum(-4096);
        x2Spinner.setMaximum(4096);
        x2Spinner.setSelection(x2);
        GridData x2SpinnerGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        x2Spinner.setLayoutData(x2SpinnerGd);

        // Y2 (negative values are relative to bottom edge)
        Label y2Label = new Label(dialog, SWT.NONE);
        y2Label.setText("Y2:");
        GridData y2LabelGd = new GridData();
        y2Label.setLayoutData(y2LabelGd);
        Spinner y2Spinner = new Spinner(dialog, SWT.BORDER);
        y2Spinner.setMinimum(-4096);
        y2Spinner.setMaximum(4096);
        y2Spinner.setSelection(y2);
        GridData y2SpinnerGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        y2Spinner.setLayoutData(y2SpinnerGd);

        // Circle Radius (only shown for circle)
        Label radiusLabel = new Label(dialog, SWT.NONE);
        radiusLabel.setText("Radius:");
        GridData radiusLabelGd = new GridData();
        radiusLabel.setLayoutData(radiusLabelGd);
        Spinner radiusSpinner = new Spinner(dialog, SWT.BORDER);
        radiusSpinner.setMinimum(1);
        radiusSpinner.setMaximum(4096);
        radiusSpinner.setSelection(circleRadius);
        GridData radiusSpinnerGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        radiusSpinner.setLayoutData(radiusSpinnerGd);

        // Initially hide/show based on shape type
        boolean isCircle = shapeIndex == 1;
        x2Label.setVisible(!isCircle);
        x2Spinner.setVisible(!isCircle);
        y2Label.setVisible(!isCircle);
        y2Spinner.setVisible(!isCircle);
        radiusLabel.setVisible(isCircle);
        radiusSpinner.setVisible(isCircle);
        if (isCircle) {
            x2LabelGd.exclude = true;
            x2SpinnerGd.exclude = true;
            y2LabelGd.exclude = true;
            y2SpinnerGd.exclude = true;
        } else {
            radiusLabelGd.exclude = true;
            radiusSpinnerGd.exclude = true;
        }

        // Update visibility when shape changes
        shapeCombo.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                boolean isCircle = shapeCombo.getSelectionIndex() == 1;
                x2Label.setVisible(!isCircle);
                x2Spinner.setVisible(!isCircle);
                y2Label.setVisible(!isCircle);
                y2Spinner.setVisible(!isCircle);
                radiusLabel.setVisible(isCircle);
                radiusSpinner.setVisible(isCircle);
                x2LabelGd.exclude = isCircle;
                x2SpinnerGd.exclude = isCircle;
                y2LabelGd.exclude = isCircle;
                y2SpinnerGd.exclude = isCircle;
                radiusLabelGd.exclude = !isCircle;
                radiusSpinnerGd.exclude = !isCircle;
                dialog.layout(true, true);
                dialog.pack();
            }
        });

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
            shapeIndex = shapeCombo.getSelectionIndex();
            x1 = x1Spinner.getSelection();
            y1 = y1Spinner.getSelection();
            x2 = x2Spinner.getSelection();
            y2 = y2Spinner.getSelection();
            circleRadius = radiusSpinner.getSelection();
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
