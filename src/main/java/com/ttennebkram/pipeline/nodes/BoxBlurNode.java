package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Box Blur (averaging) node - applies a simple normalized box filter.
 */
@NodeInfo(
    name = "BoxBlur",
    category = "Blur",
    aliases = {"Box Blur"}
)
public class BoxBlurNode extends ProcessingNode {
    private int kernelSizeX = 5; // Odd integers only
    private int kernelSizeY = 5; // Odd integers only

    public BoxBlurNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Box Blur", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSizeX() { return kernelSizeX; }
    public void setKernelSizeX(int v) { kernelSizeX = v; }
    public int getKernelSizeY() { return kernelSizeY; }
    public void setKernelSizeY(int v) { kernelSizeY = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        Mat output = new Mat();
        Imgproc.blur(input, output, new Size(kernelSizeX, kernelSizeY));
        return output;
    }

    @Override
    public String getDescription() {
        return "Box Blur\ncv2.blur(src, ksize)";
    }

    @Override
    public String getDisplayName() {
        return "Box Blur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Box Blur Properties");
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

        // Kernel Size X (odd integers only)
        new Label(dialog, SWT.NONE).setText("Kernel Size X:");
        Scale kxScale = new Scale(dialog, SWT.HORIZONTAL);
        kxScale.setMinimum(0);  // Represents 1
        kxScale.setMaximum(15); // Represents 31
        // Clamp slider position to valid range, but keep actual value
        int kxSliderPos = Math.min(Math.max(kernelSizeX / 2, 0), 15);
        kxScale.setSelection(kxSliderPos);
        kxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kxLabel = new Label(dialog, SWT.NONE);
        kxLabel.setText(String.valueOf(kernelSizeX)); // Show real value
        GridData kxLabelGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        kxLabelGd.widthHint = 25;
        kxLabel.setLayoutData(kxLabelGd);
        kxScale.addListener(SWT.Selection, e -> kxLabel.setText(String.valueOf(kxScale.getSelection() * 2 + 1)));

        // Kernel Size Y (odd integers only)
        new Label(dialog, SWT.NONE).setText("Kernel Size Y:");
        Scale kyScale = new Scale(dialog, SWT.HORIZONTAL);
        kyScale.setMinimum(0);  // Represents 1
        kyScale.setMaximum(15); // Represents 31
        // Clamp slider position to valid range, but keep actual value
        int kySliderPos = Math.min(Math.max(kernelSizeY / 2, 0), 15);
        kyScale.setSelection(kySliderPos);
        kyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kyLabel = new Label(dialog, SWT.NONE);
        kyLabel.setText(String.valueOf(kernelSizeY)); // Show real value
        GridData kyLabelGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        kyLabelGd.widthHint = 25;
        kyLabel.setLayoutData(kyLabelGd);
        kyScale.addListener(SWT.Selection, e -> kyLabel.setText(String.valueOf(kyScale.getSelection() * 2 + 1)));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            kernelSizeX = kxScale.getSelection() * 2 + 1; // Convert back to odd integer
            kernelSizeY = kyScale.getSelection() * 2 + 1; // Convert back to odd integer
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
        json.addProperty("kernelSizeX", kernelSizeX);
        json.addProperty("kernelSizeY", kernelSizeY);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("kernelSizeX")) kernelSizeX = json.get("kernelSizeX").getAsInt();
        if (json.has("kernelSizeY")) kernelSizeY = json.get("kernelSizeY").getAsInt();
    }
}
