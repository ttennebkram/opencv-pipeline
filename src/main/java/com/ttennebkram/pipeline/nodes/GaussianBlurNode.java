package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Gaussian Blur effect node.
 */
@NodeInfo(
    name = "GaussianBlur",
    category = "Blur",
    aliases = {"Gaussian Blur"}
)
public class GaussianBlurNode extends ProcessingNode {
    private int kernelSizeX = 7;
    private int kernelSizeY = 7;
    private double sigmaX = 0.0;

    public GaussianBlurNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Gaussian Blur", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSizeX() { return kernelSizeX; }
    public void setKernelSizeX(int v) { kernelSizeX = v; }
    public int getKernelSizeY() { return kernelSizeY; }
    public void setKernelSizeY(int v) { kernelSizeY = v; }
    public double getSigmaX() { return sigmaX; }
    public void setSigmaX(double v) { sigmaX = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        // Ensure odd kernel sizes
        int kx = (kernelSizeX % 2 == 0) ? kernelSizeX + 1 : kernelSizeX;
        int ky = (kernelSizeY % 2 == 0) ? kernelSizeY + 1 : kernelSizeY;

        Mat output = new Mat();
        Imgproc.GaussianBlur(input, output, new Size(kx, ky), sigmaX);
        return output;
    }

    @Override
    public String getDescription() {
        return "Gaussian Blur\ncv2.GaussianBlur(src, ksize, sigmaX)";
    }

    @Override
    public String getDisplayName() {
        return "Gaussian Blur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    protected int getPropertiesDialogColumns() {
        return 3;
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = columns;
        sep.setLayoutData(sepGd);

        // Kernel Size X
        new Label(dialog, SWT.NONE).setText("Kernel Size X:");
        Scale kxScale = new Scale(dialog, SWT.HORIZONTAL);
        kxScale.setMinimum(1);
        kxScale.setMaximum(31);
        // Clamp slider position to valid range, but keep actual value
        int kxSliderPos = Math.min(Math.max(kernelSizeX, 1), 31);
        kxScale.setSelection(kxSliderPos);
        kxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kxLabel = new Label(dialog, SWT.NONE);
        kxLabel.setText(String.valueOf(kernelSizeX)); // Show real value
        kxScale.addListener(SWT.Selection, e -> kxLabel.setText(String.valueOf(kxScale.getSelection())));

        // Kernel Size Y
        new Label(dialog, SWT.NONE).setText("Kernel Size Y:");
        Scale kyScale = new Scale(dialog, SWT.HORIZONTAL);
        kyScale.setMinimum(1);
        kyScale.setMaximum(31);
        // Clamp slider position to valid range, but keep actual value
        int kySliderPos = Math.min(Math.max(kernelSizeY, 1), 31);
        kyScale.setSelection(kySliderPos);
        kyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kyLabel = new Label(dialog, SWT.NONE);
        kyLabel.setText(String.valueOf(kernelSizeY)); // Show real value
        kyScale.addListener(SWT.Selection, e -> kyLabel.setText(String.valueOf(kyScale.getSelection())));

        // Sigma X
        new Label(dialog, SWT.NONE).setText("Sigma X:");
        Scale sigmaScale = new Scale(dialog, SWT.HORIZONTAL);
        sigmaScale.setMinimum(0);
        sigmaScale.setMaximum(100);
        // Clamp slider position to valid range, but keep actual value
        int sigmaSliderPos = Math.min(Math.max((int)(sigmaX * 10), 0), 100);
        sigmaScale.setSelection(sigmaSliderPos);
        sigmaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label sigmaLabel = new Label(dialog, SWT.NONE);
        sigmaLabel.setText(sigmaX == 0 ? "0 (auto)" : String.format("%.1f", sigmaX)); // Show real value
        sigmaScale.addListener(SWT.Selection, e -> {
            double val = sigmaScale.getSelection() / 10.0;
            sigmaLabel.setText(val == 0 ? "0 (auto)" : String.format("%.1f", val));
        });

        return () -> {
            kernelSizeX = kxScale.getSelection();
            kernelSizeY = kyScale.getSelection();
            sigmaX = sigmaScale.getSelection() / 10.0;
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSizeX", kernelSizeX);
        json.addProperty("kernelSizeY", kernelSizeY);
        json.addProperty("sigmaX", sigmaX);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("kernelSizeX")) kernelSizeX = json.get("kernelSizeX").getAsInt();
        if (json.has("kernelSizeY")) kernelSizeY = json.get("kernelSizeY").getAsInt();
        if (json.has("sigmaX")) sigmaX = json.get("sigmaX").getAsDouble();
    }
}
