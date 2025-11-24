package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * AddWeighted node - blends two images with configurable weights.
 * dst = alpha * src1 + beta * src2 + gamma
 */
public class AddWeightedNode extends ProcessingNode {
    // Second input queue for dual-input processing
    protected BlockingQueue<Mat> inputQueue2;

    // Cached last frames from each input
    private Mat lastInput1 = null;
    private Mat lastInput2 = null;

    // Blending parameters
    private double alpha = 0.5;  // Weight for first image
    private double beta = 0.5;   // Weight for second image
    private double gamma = 0.0;  // Scalar added to sum

    // Sync mode: when true, only process when BOTH queues have new input
    protected boolean queuesInSync = false;

    public boolean isQueuesInSync() {
        return queuesInSync;
    }

    public void setQueuesInSync(boolean queuesInSync) {
        this.queuesInSync = queuesInSync;
    }

    public AddWeightedNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Add Weighted", x, y);
    }

    // Getters/setters for serialization
    public double getAlpha() { return alpha; }
    public void setAlpha(double v) { alpha = v; }
    public double getBeta() { return beta; }
    public void setBeta(double v) { beta = v; }
    public double getGamma() { return gamma; }
    public void setGamma(double v) { gamma = v; }

    // Get second input connection point (below the first)
    public Point getInputPoint2() {
        return new Point(x, y + height * 3 / 4);
    }

    @Override
    public Point getInputPoint() {
        // First input point is at 1/4 height
        return new Point(x, y + height / 4);
    }

    public void setInputQueue2(BlockingQueue<Mat> queue) {
        this.inputQueue2 = queue;
    }

    public BlockingQueue<Mat> getInputQueue2() {
        return inputQueue2;
    }

    @Override
    public Mat process(Mat input) {
        // This is called with single input; for dual input we override startProcessing
        return input;
    }

    // Process two input images
    public Mat processDual(Mat input1, Mat input2) {
        if (input1 == null || input2 == null) {
            return input1 != null ? input1.clone() : (input2 != null ? input2.clone() : null);
        }

        Mat output = new Mat();

        // Resize input2 to match input1 if sizes differ
        Mat resized2 = input2;
        if (input1.width() != input2.width() || input1.height() != input2.height()) {
            resized2 = new Mat();
            org.opencv.imgproc.Imgproc.resize(input2, resized2, new Size(input1.width(), input1.height()));
        }

        // Convert to same type if needed
        Mat converted2 = resized2;
        if (input1.type() != resized2.type()) {
            converted2 = new Mat();
            resized2.convertTo(converted2, input1.type());
        }

        // dst = alpha * src1 + beta * src2 + gamma
        Core.addWeighted(input1, alpha, converted2, beta, gamma, output);

        // Clean up temporary mats
        if (resized2 != input2) resized2.release();
        if (converted2 != resized2) converted2.release();

        return output;
    }

    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    boolean gotInput1 = false;
                    boolean gotInput2 = false;

                    // Poll both queues for new frames (non-blocking)
                    if (inputQueue != null) {
                        Mat newInput1 = inputQueue.poll(10, TimeUnit.MILLISECONDS);
                        if (newInput1 != null) {
                            if (lastInput1 != null) lastInput1.release();
                            lastInput1 = newInput1;
                            gotInput1 = true;
                        }
                    }

                    if (inputQueue2 != null) {
                        Mat newInput2 = inputQueue2.poll(10, TimeUnit.MILLISECONDS);
                        if (newInput2 != null) {
                            if (lastInput2 != null) lastInput2.release();
                            lastInput2 = newInput2;
                            gotInput2 = true;
                        }
                    }

                    // Determine if we should process based on sync mode
                    boolean shouldProcess;
                    if (queuesInSync) {
                        // Sync mode: only process when BOTH queues have new input
                        // But still need both frames to exist (startup condition)
                        shouldProcess = gotInput1 && gotInput2 && lastInput1 != null && lastInput2 != null;
                    } else {
                        // Default mode: process if either queue has new input and both frames exist
                        shouldProcess = (gotInput1 || gotInput2) && lastInput1 != null && lastInput2 != null;
                    }

                    if (shouldProcess) {
                        Mat output = processDual(lastInput1, lastInput2);

                        if (output != null) {
                            setOutputMat(output);
                            notifyFrame(output);

                            if (outputQueue != null) {
                                outputQueue.put(output);
                            }
                        }
                    } else if (!gotInput1 && !gotInput2) {
                        // Small sleep if no input available
                        Thread.sleep(10);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "Processing-" + name + "-Thread");
        processingThread.start();
    }

    @Override
    public void stopProcessing() {
        super.stopProcessing();
        // Clear cached frames
        if (lastInput1 != null) {
            lastInput1.release();
            lastInput1 = null;
        }
        if (lastInput2 != null) {
            lastInput2.release();
            lastInput2 = null;
        }
        // Clear second queue
        if (inputQueue2 != null) {
            inputQueue2.clear();
        }
    }

    @Override
    public void paint(GC gc) {
        // Draw node background
        gc.setBackground(new Color(255, 230, 200)); // Light orange for arithmetic
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(200, 100, 0));
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(name, x + 10, y + 5, true);
        boldFont.dispose();

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 25;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 40, true);
        }

        // Draw connection points (custom for dual input)
        drawDualInputConnectionPoints(gc);
    }

    protected void drawDualInputConnectionPoints(GC gc) {
        int radius = 6;

        // Draw first input point (top left)
        Point input1 = getInputPoint();
        gc.setBackground(new Color(200, 220, 255));
        gc.fillOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(70, 100, 180));
        gc.setLineWidth(2);
        gc.drawOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);

        // Draw second input point (bottom left)
        Point input2 = getInputPoint2();
        gc.setBackground(new Color(200, 220, 255));
        gc.fillOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(70, 100, 180));
        gc.drawOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);

        // Draw output point on right side
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200));
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);
    }

    @Override
    public String getDescription() {
        return "Blend Images with Weights\ncv2.addWeighted(src1, alpha, src2, beta, gamma)";
    }

    @Override
    public String getDisplayName() {
        return "Add Weighted";
    }

    @Override
    public String getCategory() {
        return "Dual Input Nodes";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Add Weighted Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Description
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

        // Alpha (weight for first image)
        new Label(dialog, SWT.NONE).setText("Alpha (img1 weight):");
        Scale alphaScale = new Scale(dialog, SWT.HORIZONTAL);
        alphaScale.setMinimum(0);
        alphaScale.setMaximum(100);
        alphaScale.setSelection((int)(alpha * 100));
        alphaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label alphaLabel = new Label(dialog, SWT.NONE);
        alphaLabel.setText(String.format("%.2f", alpha));
        alphaScale.addListener(SWT.Selection, e -> {
            double val = alphaScale.getSelection() / 100.0;
            alphaLabel.setText(String.format("%.2f", val));
        });

        // Beta (weight for second image)
        new Label(dialog, SWT.NONE).setText("Beta (img2 weight):");
        Scale betaScale = new Scale(dialog, SWT.HORIZONTAL);
        betaScale.setMinimum(0);
        betaScale.setMaximum(100);
        betaScale.setSelection((int)(beta * 100));
        betaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label betaLabel = new Label(dialog, SWT.NONE);
        betaLabel.setText(String.format("%.2f", beta));
        betaScale.addListener(SWT.Selection, e -> {
            double val = betaScale.getSelection() / 100.0;
            betaLabel.setText(String.format("%.2f", val));
        });

        // Gamma (scalar offset)
        new Label(dialog, SWT.NONE).setText("Gamma (offset):");
        Scale gammaScale = new Scale(dialog, SWT.HORIZONTAL);
        gammaScale.setMinimum(0);
        gammaScale.setMaximum(255);
        gammaScale.setSelection((int)gamma);
        gammaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label gammaLabel = new Label(dialog, SWT.NONE);
        gammaLabel.setText(String.valueOf((int)gamma));
        gammaScale.addListener(SWT.Selection, e -> gammaLabel.setText(String.valueOf(gammaScale.getSelection())));

        // Queues In Sync checkbox
        Button syncCheckbox = new Button(dialog, SWT.CHECK);
        syncCheckbox.setText("Queues In Sync");
        syncCheckbox.setSelection(queuesInSync);
        syncCheckbox.setToolTipText("When checked, only process when both inputs receive new frames");
        GridData syncGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        syncGd.horizontalSpan = 3;
        syncCheckbox.setLayoutData(syncGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            alpha = alphaScale.getSelection() / 100.0;
            beta = betaScale.getSelection() / 100.0;
            gamma = gammaScale.getSelection();
            queuesInSync = syncCheckbox.getSelection();
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
