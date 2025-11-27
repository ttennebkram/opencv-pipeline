package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Template Matching node - finds where a template image appears in a source image.
 * Input 1: Source image
 * Input 2: Template image
 * Output: Result matrix showing correlation/matching scores at each location
 */
@NodeInfo(
    name = "MatchTemplate",
    category = "Detection",
    aliases = {"Match Template"}
)
public class MatchTemplateNode extends DualInputNode {
    private int method = Imgproc.TM_CCOEFF_NORMED; // Default matching method
    private int outputMode = 0; // 0=result only, 1=source+rectangle, 2=source+overlay
    private int rectColorR = 0;   // Green rectangle by default
    private int rectColorG = 255;
    private int rectColorB = 0;
    private int rectThickness = 5; // Default thickness for visibility in thumbnails

    private static final String[] METHOD_NAMES = {
        "TM_SQDIFF", "TM_SQDIFF_NORMED",
        "TM_CCORR", "TM_CCORR_NORMED",
        "TM_CCOEFF", "TM_CCOEFF_NORMED"
    };
    private static final int[] METHOD_VALUES = {
        Imgproc.TM_SQDIFF, Imgproc.TM_SQDIFF_NORMED,
        Imgproc.TM_CCORR, Imgproc.TM_CCORR_NORMED,
        Imgproc.TM_CCOEFF, Imgproc.TM_CCOEFF_NORMED
    };
    private static final String[] OUTPUT_MODE_NAMES = {
        "Result Matrix Only",
        "Source + Rectangle",
        "Source + Result Overlay"
    };

    public MatchTemplateNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Match Template", x, y);
    }

    @Override
    public Mat process(Mat input) {
        // This is called with single input; for dual input we use processDual
        return input;
    }

    @Override
    public Mat processDual(Mat source, Mat template) {
        if (source == null || template == null) {
            return source != null ? source.clone() : null;
        }

        // Template must be smaller than source
        if (template.width() >= source.width() || template.height() >= source.height()) {
            return source.clone();
        }

        // Template must have at least 1x1 size
        if (template.width() < 1 || template.height() < 1) {
            return source.clone();
        }

        Mat result = new Mat();

        try {
            // Perform template matching - returns correlation matrix (CV_32F)
            Imgproc.matchTemplate(source, template, result, method);

            // Find the best match location for debug output
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

            // For TM_SQDIFF and TM_SQDIFF_NORMED, best match is minimum; for others, maximum
            Point matchLoc;
            double matchValue;
            if (method == Imgproc.TM_SQDIFF || method == Imgproc.TM_SQDIFF_NORMED) {
                matchLoc = mmr.minLoc;
                matchValue = mmr.minVal;
            } else {
                matchLoc = mmr.maxLoc;
                matchValue = mmr.maxVal;
            }

            // Handle output mode
            if (outputMode == 1) {
                // Mode 1: Source + Rectangle
                Mat output = source.clone();

                // Calculate rectangle coordinates
                int x1 = (int) Math.max(0, matchLoc.x);
                int y1 = (int) Math.max(0, matchLoc.y);
                int x2 = (int) Math.min(source.width() - 1, matchLoc.x + template.cols());
                int y2 = (int) Math.min(source.height() - 1, matchLoc.y + template.rows());

                // Draw rectangle around the matched region
                Point topLeft = new Point(x1, y1);
                Point bottomRight = new Point(x2, y2);
                Imgproc.rectangle(output, topLeft, bottomRight,
                                new Scalar(rectColorB, rectColorG, rectColorR), rectThickness);

                result.release();
                return output;
            } else if (outputMode == 2) {
                // Mode 2: Source + Result Overlay
                // Normalize result to 8-bit
                Mat normalized = new Mat();
                Core.normalize(result, normalized, 0, 255, Core.NORM_MINMAX);
                Mat result8u = new Mat();
                normalized.convertTo(result8u, org.opencv.core.CvType.CV_8UC1);

                // Resize result to match source dimensions
                Mat resizedResult = new Mat();
                Imgproc.resize(result8u, resizedResult, new org.opencv.core.Size(source.width(), source.height()));

                // Convert result to color (for blending)
                Mat resultColor = new Mat();
                Imgproc.cvtColor(resizedResult, resultColor, Imgproc.COLOR_GRAY2BGR);

                // Apply colormap to result for better visualization
                Imgproc.applyColorMap(resizedResult, resultColor, Imgproc.COLORMAP_JET);

                // Blend with source image
                Mat output = new Mat();
                Core.addWeighted(source, 0.6, resultColor, 0.4, 0, output);

                result.release();
                normalized.release();
                result8u.release();
                resizedResult.release();
                resultColor.release();

                return output;
            } else {
                // Mode 0: Result Matrix Only (default)
                Mat normalized = new Mat();
                Core.normalize(result, normalized, 0, 255, Core.NORM_MINMAX);

                Mat result8u = new Mat();
                normalized.convertTo(result8u, org.opencv.core.CvType.CV_8UC1);

                result.release();
                normalized.release();

                // Return the 8-bit result matrix for visualization
                return result8u;
            }
        } catch (Exception e) {
            System.err.println("MatchTemplate error: " + e.getMessage());
            e.printStackTrace();
            // Return source image on error
            if (!result.empty()) {
                result.release();
            }
            return source.clone();
        }
    }

    @Override
    public void paint(GC gc) {
        // Draw node background
        Color bgColor = new Color(255, 240, 220); // Light peach for template matching
        gc.setBackground(bgColor);
        gc.fillRoundRectangle(x, y, width, height, 10, 10);
        bgColor.dispose();

        // Draw border
        Color borderColor = new Color(200, 140, 80);
        gc.setForeground(borderColor);
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);
        borderColor.dispose();

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(name, x + 10, y + 5, true);
        boldFont.dispose();

        // Draw thread priority label
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        // Red text if priority is below 5, otherwise dark gray
        int currentPriority = getThreadPriority();
        if (currentPriority < 5) {
            Color redColor = new Color(200, 0, 0); // Red for low priority
            gc.setForeground(redColor);
            gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
            redColor.dispose();
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
            gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
        }
        smallFont.dispose();

        // Draw thumbnail if available (centered horizontally)
        Rectangle bounds = getThumbnailBounds();
        if (bounds != null) {
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 35;
            drawThumbnail(gc, thumbX, thumbY);
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 40, true);
        }

        // Draw connection points (custom for dual input)
        drawDualInputConnectionPoints(gc);
    }

    protected void drawDualInputConnectionPoints(GC gc) {
        int radius = 6;

        // Draw first input point (top left) - Source image
        org.eclipse.swt.graphics.Point input1 = getInputPoint();
        Color input1BgColor = new Color(200, 220, 255);
        gc.setBackground(input1BgColor);
        gc.fillOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        input1BgColor.dispose();
        Color input1FgColor = new Color(70, 100, 180);
        gc.setForeground(input1FgColor);
        gc.setLineWidth(2);
        gc.drawOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        input1FgColor.dispose();

        // Draw second input point (bottom left) - Template image
        org.eclipse.swt.graphics.Point input2 = getInputPoint2();
        Color input2BgColor = new Color(200, 220, 255);
        gc.setBackground(input2BgColor);
        gc.fillOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        input2BgColor.dispose();
        Color input2FgColor = new Color(70, 100, 180);
        gc.setForeground(input2FgColor);
        gc.drawOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        input2FgColor.dispose();

        // Draw output point on right side
        org.eclipse.swt.graphics.Point output = getOutputPoint();
        Color outputBgColor = new Color(220, 255, 220);
        gc.setBackground(outputBgColor);
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        outputBgColor.dispose();
        Color outputFgColor = new Color(80, 180, 80);
        gc.setForeground(outputFgColor);
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        outputFgColor.dispose();
        gc.setLineWidth(1);
    }

    @Override
    public String getDescription() {
        return "Template Matching\nresult = cv2.matchTemplate(source, template, method)";
    }

    @Override
    public String getDisplayName() {
        return "Match Template";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    public int getMethod() {
        return method;
    }

    public void setMethod(int method) {
        this.method = method;
    }

    public int getOutputMode() {
        return outputMode;
    }

    public void setOutputMode(int outputMode) {
        this.outputMode = outputMode;
    }

    public int getRectColorR() {
        return rectColorR;
    }

    public void setRectColorR(int r) {
        this.rectColorR = r;
    }

    public int getRectColorG() {
        return rectColorG;
    }

    public void setRectColorG(int g) {
        this.rectColorG = g;
    }

    public int getRectColorB() {
        return rectColorB;
    }

    public void setRectColorB(int b) {
        this.rectColorB = b;
    }

    public int getRectThickness() {
        return rectThickness;
    }

    public void setRectThickness(int thickness) {
        this.rectThickness = thickness;
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Match Template Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription() + "\n\nInput 1: Source image\nInput 2: Template to find\nOutput: Correlation matrix");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Method selection
        new Label(dialog, SWT.NONE).setText("Method:");
        Combo methodCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        methodCombo.setItems(METHOD_NAMES);
        // Find current method index
        int currentIndex = 5; // Default to TM_CCOEFF_NORMED
        for (int i = 0; i < METHOD_VALUES.length; i++) {
            if (METHOD_VALUES[i] == method) {
                currentIndex = i;
                break;
            }
        }
        methodCombo.select(currentIndex);
        GridData methodGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        methodCombo.setLayoutData(methodGd);

        // Output Mode selection
        new Label(dialog, SWT.NONE).setText("Output Mode:");
        Combo outputModeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        outputModeCombo.setItems(OUTPUT_MODE_NAMES);
        outputModeCombo.select(outputMode);
        GridData outputModeGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        outputModeCombo.setLayoutData(outputModeGd);

        // Queues In Sync checkbox
        new Label(dialog, SWT.NONE).setText("Queues In Sync:");
        Button syncCheckbox = new Button(dialog, SWT.CHECK);
        syncCheckbox.setSelection(queuesInSync);
        syncCheckbox.setToolTipText("When checked, only process when both inputs receive new frames");

        // Rectangle Color inputs
        new Label(dialog, SWT.NONE).setText("Rectangle Color (R G B):");
        Composite colorComp = new Composite(dialog, SWT.NONE);
        colorComp.setLayout(new GridLayout(3, false));
        colorComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Text rText = new Text(colorComp, SWT.BORDER);
        rText.setText(String.valueOf(rectColorR));
        rText.setLayoutData(new GridData(40, SWT.DEFAULT));

        Text gText = new Text(colorComp, SWT.BORDER);
        gText.setText(String.valueOf(rectColorG));
        gText.setLayoutData(new GridData(40, SWT.DEFAULT));

        Text bText = new Text(colorComp, SWT.BORDER);
        bText.setText(String.valueOf(rectColorB));
        bText.setLayoutData(new GridData(40, SWT.DEFAULT));

        // Rectangle Thickness input
        new Label(dialog, SWT.NONE).setText("Rectangle Thickness:");
        Text thicknessText = new Text(dialog, SWT.BORDER);
        thicknessText.setText(String.valueOf(rectThickness));
        thicknessText.setLayoutData(new GridData(60, SWT.DEFAULT));

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
            method = METHOD_VALUES[methodCombo.getSelectionIndex()];
            outputMode = outputModeCombo.getSelectionIndex();
            queuesInSync = syncCheckbox.getSelection();
            try {
                rectColorR = Math.max(0, Math.min(255, Integer.parseInt(rText.getText())));
                rectColorG = Math.max(0, Math.min(255, Integer.parseInt(gText.getText())));
                rectColorB = Math.max(0, Math.min(255, Integer.parseInt(bText.getText())));
                rectThickness = Math.max(1, Math.min(50, Integer.parseInt(thicknessText.getText())));
            } catch (NumberFormatException ex) {
                // Keep existing values if parse fails
            }
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
        json.addProperty("method", method);
        json.addProperty("queuesInSync", queuesInSync);
        json.addProperty("outputMode", outputMode);
        json.addProperty("rectColorR", rectColorR);
        json.addProperty("rectColorG", rectColorG);
        json.addProperty("rectColorB", rectColorB);
        json.addProperty("rectThickness", rectThickness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("method")) {
            method = json.get("method").getAsInt();
        }
        if (json.has("queuesInSync")) {
            queuesInSync = json.get("queuesInSync").getAsBoolean();
        }
        if (json.has("outputMode")) {
            outputMode = json.get("outputMode").getAsInt();
        }
        if (json.has("rectColorR")) {
            rectColorR = json.get("rectColorR").getAsInt();
        }
        if (json.has("rectColorG")) {
            rectColorG = json.get("rectColorG").getAsInt();
        }
        if (json.has("rectColorB")) {
            rectColorB = json.get("rectColorB").getAsInt();
        }
        if (json.has("rectThickness")) {
            rectThickness = json.get("rectThickness").getAsInt();
        }
    }
}
