package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Text node - draws text on the image.
 */
public class TextNode extends ProcessingNode {
    private static final String[] FONT_NAMES = {
        "Simplex", "Plain", "Duplex", "Complex", "Triplex",
        "Complex Small", "Script Simplex", "Script Complex"
    };
    private static final int[] FONT_CODES = {
        Imgproc.FONT_HERSHEY_SIMPLEX,
        Imgproc.FONT_HERSHEY_PLAIN,
        Imgproc.FONT_HERSHEY_DUPLEX,
        Imgproc.FONT_HERSHEY_COMPLEX,
        Imgproc.FONT_HERSHEY_TRIPLEX,
        Imgproc.FONT_HERSHEY_COMPLEX_SMALL,
        Imgproc.FONT_HERSHEY_SCRIPT_SIMPLEX,
        Imgproc.FONT_HERSHEY_SCRIPT_COMPLEX
    };

    private String text = "Hello";
    private int posX = 50, posY = 100;
    private int fontIndex = 0;
    private double fontScale = 1.0;
    private int colorR = 0, colorG = 255, colorB = 0; // Green default
    private int thickness = 2;
    private boolean bold = false;
    private boolean italic = false;

    public TextNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Text", x, y);
    }

    // Getters/setters for serialization
    public String getText() { return text; }
    public void setText(String v) { text = v; }
    public int getPosX() { return posX; }
    public void setPosX(int v) { posX = v; }
    public int getPosY() { return posY; }
    public void setPosY(int v) { posY = v; }
    public int getFontIndex() { return fontIndex; }
    public void setFontIndex(int v) { fontIndex = v; }
    public double getFontScale() { return fontScale; }
    public void setFontScale(double v) { fontScale = v; }
    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }
    public int getThickness() { return thickness; }
    public void setThickness(int v) { thickness = v; }
    public boolean isBold() { return bold; }
    public void setBold(boolean v) { bold = v; }
    public boolean isItalic() { return italic; }
    public void setItalic(boolean v) { italic = v; }

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

        int font = FONT_CODES[fontIndex];
        if (italic) {
            font |= Imgproc.FONT_ITALIC;
        }

        // Convert coordinates (negative values are relative to right/bottom)
        int absX = toAbsoluteX(posX, input.width());
        int absY = toAbsoluteY(posY, input.height());

        // Bold effect: increase thickness
        int effectiveThickness = bold ? thickness + 1 : thickness;

        Imgproc.putText(output, text, new org.opencv.core.Point(absX, absY),
            font, fontScale, color, effectiveThickness);

        return output;
    }

    @Override
    public String getDescription() {
        return "Draw Text\ncv2.putText(img, text, org, font, scale, color, thickness)";
    }

    @Override
    public String getDisplayName() {
        return "Text";
    }

    @Override
    public String getCategory() {
        return "Content";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Text Properties");
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

        // Text
        new Label(dialog, SWT.NONE).setText("Text:");
        Text textField = new Text(dialog, SWT.BORDER);
        textField.setText(text);
        textField.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Position X (negative values are relative to right edge)
        new Label(dialog, SWT.NONE).setText("X:");
        Spinner xSpinner = new Spinner(dialog, SWT.BORDER);
        xSpinner.setMinimum(-4096);
        xSpinner.setMaximum(4096);
        xSpinner.setSelection(posX);
        xSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Position Y (negative values are relative to bottom edge)
        new Label(dialog, SWT.NONE).setText("Y:");
        Spinner ySpinner = new Spinner(dialog, SWT.BORDER);
        ySpinner.setMinimum(-4096);
        ySpinner.setMaximum(4096);
        ySpinner.setSelection(posY);
        ySpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Font
        new Label(dialog, SWT.NONE).setText("Font:");
        Combo fontCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        fontCombo.setItems(FONT_NAMES);
        fontCombo.select(fontIndex);
        fontCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Font Scale
        new Label(dialog, SWT.NONE).setText("Scale:");
        Spinner scaleSpinner = new Spinner(dialog, SWT.BORDER);
        scaleSpinner.setMinimum(1);
        scaleSpinner.setMaximum(100);
        scaleSpinner.setDigits(1);
        scaleSpinner.setSelection((int)(fontScale * 10));
        scaleSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

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
        thickSpinner.setMaximum(20);
        thickSpinner.setSelection(thickness);
        thickSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Bold checkbox
        new Label(dialog, SWT.NONE).setText("Bold:");
        Button boldCheck = new Button(dialog, SWT.CHECK);
        boldCheck.setSelection(bold);

        // Italic checkbox
        new Label(dialog, SWT.NONE).setText("Italic:");
        Button italicCheck = new Button(dialog, SWT.CHECK);
        italicCheck.setSelection(italic);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            text = textField.getText();
            posX = xSpinner.getSelection();
            posY = ySpinner.getSelection();
            fontIndex = fontCombo.getSelectionIndex();
            fontScale = scaleSpinner.getSelection() / 10.0;
            colorR = rSpinner.getSelection();
            colorG = gSpinner.getSelection();
            colorB = bSpinner.getSelection();
            thickness = thickSpinner.getSelection();
            bold = boldCheck.getSelection();
            italic = italicCheck.getSelection();
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
}
