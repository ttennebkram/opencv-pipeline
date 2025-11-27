package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Blank source node - generates a solid color (default black) image.
 */
@NodeInfo(name = "BlankSource", category = "Source", aliases = {"Blank Source"})
public class BlankSourceNode extends SourceNode {
    private int imageWidth = 640;
    private int imageHeight = 480;
    private int colorIndex = 0; // Default to black
    private int fpsIndex = 0; // Default to 1 fps
    private Mat blankImage = null;

    // Color presets: name, R, G, B
    private static final String[] COLOR_NAMES = {"Black", "White", "Red", "Green", "Blue", "Yellow"};
    private static final int[][] COLOR_VALUES = {
        {0, 0, 0},       // Black
        {255, 255, 255}, // White
        {255, 0, 0},     // Red
        {0, 255, 0},     // Green
        {0, 0, 255},     // Blue
        {255, 255, 0}    // Yellow
    };

    // FPS presets
    private static final String[] FPS_NAMES = {"1 fps", "15 fps", "30 fps", "60 fps"};
    private static final double[] FPS_VALUES = {1.0, 15.0, 30.0, 60.0};

    // Source node height to match FileSourceNode and WebcamSourceNode
    private static final int SOURCE_NODE_HEIGHT = 120;

    public BlankSourceNode(Shell shell, Display display, int x, int y) {
        this.shell = shell;
        this.display = display;
        this.x = x;
        this.y = y;
        this.height = SOURCE_NODE_HEIGHT;
        createBlankImage();
    }

    // Getters/setters for serialization
    public int getImageWidth() { return imageWidth; }
    public void setImageWidth(int v) { imageWidth = v; createBlankImage(); }
    public int getImageHeight() { return imageHeight; }
    public void setImageHeight(int v) { imageHeight = v; createBlankImage(); }
    public int getColorIndex() { return colorIndex; }
    public void setColorIndex(int v) { colorIndex = Math.max(0, Math.min(v, COLOR_NAMES.length - 1)); createBlankImage(); }
    public int getFpsIndex() { return fpsIndex; }
    public void setFpsIndex(int v) { fpsIndex = Math.max(0, Math.min(v, FPS_VALUES.length - 1)); }

    private int getRed() { return COLOR_VALUES[colorIndex][0]; }
    private int getGreen() { return COLOR_VALUES[colorIndex][1]; }
    private int getBlue() { return COLOR_VALUES[colorIndex][2]; }

    private void createBlankImage() {
        if (blankImage != null) {
            blankImage.release();
        }
        blankImage = new Mat(imageHeight, imageWidth, CvType.CV_8UC3,
            new Scalar(getBlue(), getGreen(), getRed()));
        // Update thumbnail to show the blank image (clone so blankImage isn't released)
        setOutputMat(blankImage.clone());
    }

    public Mat getNextFrame() {
        if (blankImage == null) {
            createBlankImage();
        }
        return blankImage.clone();
    }

    public double getFps() {
        return FPS_VALUES[fpsIndex];
    }

    @Override
    public void paint(GC gc) {
        // Draw node background
        gc.setBackground(new Color(230, 240, 255));
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(0, 0, 139));
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString("Blank Source", x + 10, y + 4, true);
        boldFont.dispose();

        // Draw thread priority, work units, and FPS stats line
        drawFpsStatsLine(gc, x + 10, y + 19);

        // Draw thumbnail if available (centered horizontally)
        Rectangle bounds = getThumbnailBounds();
        if (bounds != null) {
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 34;
            drawThumbnail(gc, thumbX, thumbY);
        } else {
            // Draw placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 50, true);
        }

        // Draw connection points (output only - this is a source node)
        drawConnectionPoints(gc);
    }

    @Override
    protected void drawConnectionPoints(GC gc) {
        // Only has output point (it's a source)
        int radius = 6;
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200)); // Light orange fill
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));  // Orange border
        gc.setLineWidth(2);
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);  // Reset line width
    }

    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Blank Source Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText("Generate Solid Color Image\nnp.full((height, width, 3), color, dtype=np.uint8)");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Width
        new Label(dialog, SWT.NONE).setText("Width:");
        Spinner widthSpinner = new Spinner(dialog, SWT.BORDER);
        widthSpinner.setMinimum(1);
        widthSpinner.setMaximum(4096);
        widthSpinner.setSelection(imageWidth);
        widthSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Height
        new Label(dialog, SWT.NONE).setText("Height:");
        Spinner heightSpinner = new Spinner(dialog, SWT.BORDER);
        heightSpinner.setMinimum(1);
        heightSpinner.setMaximum(4096);
        heightSpinner.setSelection(imageHeight);
        heightSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Color dropdown
        new Label(dialog, SWT.NONE).setText("Color:");
        Combo colorCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        colorCombo.setItems(COLOR_NAMES);
        colorCombo.select(colorIndex);
        colorCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // FPS dropdown
        new Label(dialog, SWT.NONE).setText("FPS:");
        Combo fpsCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        fpsCombo.setItems(FPS_NAMES);
        fpsCombo.select(fpsIndex);
        fpsCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

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
            imageWidth = widthSpinner.getSelection();
            imageHeight = heightSpinner.getSelection();
            colorIndex = colorCombo.getSelectionIndex();
            fpsIndex = fpsCombo.getSelectionIndex();
            createBlankImage();
            dialog.dispose();
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
    public void dispose() {
        if (blankImage != null) {
            blankImage.release();
        }
    }

    // startProcessing() inherited from SourceNode

    public void serializeProperties(JsonObject json) {
        json.addProperty("imageWidth", imageWidth);
        json.addProperty("imageHeight", imageHeight);
        json.addProperty("colorIndex", colorIndex);
    }

    public void deserializeProperties(JsonObject json) {
        if (json.has("imageWidth")) imageWidth = json.get("imageWidth").getAsInt();
        if (json.has("imageHeight")) imageHeight = json.get("imageHeight").getAsInt();
        if (json.has("colorIndex")) colorIndex = json.get("colorIndex").getAsInt();
    }
}
