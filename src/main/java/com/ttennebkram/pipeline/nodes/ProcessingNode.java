package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.File;

/**
 * Base class for processing nodes with properties dialog support.
 */
public abstract class ProcessingNode extends PipelineNode {
    protected String name;

    @Override
    public String getNodeName() {
        return name;
    }
    protected Shell shell;
    protected boolean enabled = true;
    protected Runnable onChanged;  // Callback when properties change

    // Checkbox dimensions for enabled toggle
    protected static final int CHECKBOX_SIZE = 12;
    protected static final int CHECKBOX_MARGIN = 5;

    public ProcessingNode(Display display, Shell shell, String name, int x, int y) {
        this.display = display;
        this.shell = shell;
        this.name = name;
        this.x = x;
        this.y = y;
    }

    public void setOnChanged(Runnable onChanged) {
        this.onChanged = onChanged;
    }

    public Shell getShell() {
        return shell;
    }

    public void setShell(Shell shell) {
        this.shell = shell;
    }

    protected void notifyChanged() {
        if (onChanged != null) {
            onChanged.run();
        }
    }

    /**
     * Helper to add a Type label and Name field to a properties dialog.
     * Call this at the start of showPropertiesDialog() after creating the dialog.
     * @param dialog The dialog shell
     * @param columns The number of columns in the dialog's GridLayout
     * @return The Text widget for the name field (save reference to get value in OK handler)
     */
    protected Text addNameField(Shell dialog, int columns) {
        // Type label at very top (read-only)
        Label typeLabel = new Label(dialog, SWT.NONE);
        typeLabel.setText("Type:");
        Label typeValue = new Label(dialog, SWT.NONE);
        typeValue.setText(getClass().getSimpleName());
        GridData typeGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        if (columns > 2) {
            typeGd.horizontalSpan = columns - 1;
        }
        typeValue.setLayoutData(typeGd);

        // Name field
        new Label(dialog, SWT.NONE).setText("Name:");
        Text nameText = new Text(dialog, SWT.BORDER);
        nameText.setText(getDisplayLabel());
        GridData nameGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        if (columns > 2) {
            nameGd.horizontalSpan = columns - 1;
        }
        nameText.setLayoutData(nameGd);
        return nameText;
    }

    /**
     * Helper to save the name from the name field in the OK handler.
     * Call this at the start of the OK button listener.
     * @param nameText The Text widget returned by addNameField()
     */
    protected void saveNameField(Text nameText) {
        setCustomName(nameText.getText().trim());
    }

    /**
     * Template method for showing properties dialog with automatic name field handling.
     * Creates a dialog with:
     * 1. Name field at the top
     * 2. Custom content from subclass (via addPropertiesContent)
     * 3. OK/Cancel buttons that automatically save the name
     *
     * Subclasses should override addPropertiesContent() instead of showPropertiesDialog()
     * to add their specific fields.
     *
     * @param title The dialog title
     * @param columns Number of columns for the GridLayout
     */
    protected void showPropertiesDialogWithName(String title, int columns) {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText(title);
        dialog.setLayout(new GridLayout(columns, false));

        // Type label and name field (addNameField adds both)
        Text nameText = addNameField(dialog, columns);

        // Let subclass add its content
        Runnable onOk = addPropertiesContent(dialog, columns);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = columns;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            saveNameField(nameText);
            if (onOk != null) {
                onOk.run();
            }
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

    /**
     * Override this method to add custom properties content to the dialog.
     * The name field is already added at the top by showPropertiesDialogWithName().
     *
     * @param dialog The dialog shell to add content to
     * @param columns The number of columns in the dialog's GridLayout
     * @return A Runnable to execute when OK is clicked (to save properties), or null
     */
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Default implementation adds just a description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);
        return null;
    }

    // Process input Mat and return output Mat
    public abstract Mat process(Mat input);

    /**
     * Show properties dialog with name field and node-specific content.
     * Subclasses should NOT override this - instead override:
     * - getPropertiesDialogTitle() to customize the dialog title
     * - getPropertiesDialogColumns() to change the number of columns (default 2)
     * - addPropertiesContent() to add node-specific fields
     */
    public void showPropertiesDialog() {
        showPropertiesDialogWithName(getPropertiesDialogTitle(), getPropertiesDialogColumns());
    }

    /**
     * Override to customize the properties dialog title.
     * Default is "NodeDisplayName Properties"
     */
    protected String getPropertiesDialogTitle() {
        return getDisplayName() + " Properties";
    }

    /**
     * Override to change the number of columns in the properties dialog.
     * Default is 2 columns.
     */
    protected int getPropertiesDialogColumns() {
        return 2;
    }

    // Get description for tooltip
    public abstract String getDescription();

    // Get display name for toolbar button (longer, more descriptive)
    public abstract String getDisplayName();

    // Get category for toolbar grouping (e.g., "Basic", "Blur", "Edge Detection")
    public abstract String getCategory();

    @Override
    public void paint(GC gc) {
        // Draw node background - light gray if disabled
        Color bgColor = enabled ? getBackgroundColor() : new Color(DISABLED_BG_R, DISABLED_BG_G, DISABLED_BG_B);
        gc.setBackground(bgColor);
        gc.fillRoundRectangle(x, y, width, height, 10, 10);
        bgColor.dispose();

        // Draw border
        Color borderColor = getBorderColor();
        gc.setForeground(borderColor);
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);
        borderColor.dispose();

        // Draw enabled checkbox
        drawEnabledCheckbox(gc);

        // Draw title (use custom name if set, otherwise default node name) - shifted right for checkbox
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(getDisplayLabel(), x + CHECKBOX_MARGIN + CHECKBOX_SIZE + 5, y + 5, true);
        boldFont.dispose();

        // Draw thread priority label
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        // Red text if priority is below 5, otherwise dark gray
        int currentPriority = getThreadPriority();
        if (currentPriority < 5) {
            gc.setForeground(new Color(200, 0, 0)); // Red for low priority
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        }
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
        smallFont.dispose();

        // Draw input read counts on the left side
        Font tinyFont = new Font(display, "Arial", 7, SWT.NORMAL);
        gc.setFont(tinyFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        int statsX = x + 5;
        // Display input counts, wrapping to next line if 4+ digits
        if (inputReads1 >= 1000) {
            gc.drawString("In1:", statsX, y + 40, true);
            gc.drawString(formatNumber(inputReads1), statsX, y + 50, true);
        } else {
            gc.drawString("In1:" + formatNumber(inputReads1), statsX, y + 40, true);
        }
        if (hasDualInput()) {
            if (inputReads2 >= 1000) {
                gc.drawString("In2:", statsX, y + 70, true);
                gc.drawString(formatNumber(inputReads2), statsX, y + 80, true);
            } else {
                gc.drawString("In2:" + formatNumber(inputReads2), statsX, y + 70, true);
            }
        }
        tinyFont.dispose();

        // Draw thumbnail if available (to the right of stats)
        Rectangle bounds = getThumbnailBounds();
        if (bounds != null) {
            int thumbX = x + 40; // Offset to make room for stats on left
            int thumbY = y + 35;
            drawThumbnail(gc, thumbX, thumbY);
        } else {
            // Draw placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 45, y + 50, true);
        }

        // Draw connection points
        drawConnectionPoints(gc);
    }

    public String getName() {
        return name;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    /**
     * Get the bounds of the enabled checkbox for hit testing.
     * @return Rectangle with the checkbox bounds in canvas coordinates
     */
    public Rectangle getEnabledCheckboxBounds() {
        return new Rectangle(x + CHECKBOX_MARGIN, y + CHECKBOX_MARGIN, CHECKBOX_SIZE, CHECKBOX_SIZE);
    }

    /**
     * Check if a point is within the enabled checkbox.
     * @param p Point to test in canvas coordinates
     * @return true if point is within checkbox bounds
     */
    public boolean isOnEnabledCheckbox(Point p) {
        Rectangle bounds = getEnabledCheckboxBounds();
        return p.x >= bounds.x && p.x <= bounds.x + bounds.width &&
               p.y >= bounds.y && p.y <= bounds.y + bounds.height;
    }

    /**
     * Toggle the enabled state of this node.
     */
    public void toggleEnabled() {
        this.enabled = !this.enabled;
        notifyChanged();
    }

    /**
     * Draw the enabled checkbox in the top-left corner of the node.
     * Called by paint() - subclasses that override paint() should call this.
     */
    protected void drawEnabledCheckbox(GC gc) {
        Rectangle bounds = getEnabledCheckboxBounds();

        // Draw checkbox background
        gc.setBackground(display.getSystemColor(SWT.COLOR_WHITE));
        gc.fillRectangle(bounds.x, bounds.y, bounds.width, bounds.height);

        // Draw checkbox border
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        gc.setLineWidth(1);
        gc.drawRectangle(bounds.x, bounds.y, bounds.width, bounds.height);

        // Draw checkmark if enabled
        if (enabled) {
            gc.setForeground(new Color(0, 128, 0)); // Dark green checkmark
            gc.setLineWidth(2);
            // Draw checkmark as two lines
            int cx = bounds.x + bounds.width / 2;
            int cy = bounds.y + bounds.height / 2;
            // Short line from bottom-left to center-bottom
            gc.drawLine(bounds.x + 2, cy, cx - 1, bounds.y + bounds.height - 2);
            // Long line from center-bottom to top-right
            gc.drawLine(cx - 1, bounds.y + bounds.height - 2, bounds.x + bounds.width - 2, bounds.y + 2);
            gc.setLineWidth(1);
        }
    }

    public Display getDisplay() {
        return display;
    }

    // Save thumbnail to cache directory
    @Override
    public void saveThumbnailToCache(String cacheDir, int nodeIndex) {
        // Get a thread-safe clone of the output mat
        Mat matClone = getOutputMatClone();
        if (matClone != null) {
            try {
                File cacheFolder = new File(cacheDir);
                if (!cacheFolder.exists()) {
                    cacheFolder.mkdirs();
                }
                String thumbPath = cacheDir + File.separator + "node_" + nodeIndex + "_thumb.png";
                // Save the output mat as thumbnail
                Mat resized = new Mat();
                double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / matClone.width(),
                                        (double) PROCESSING_NODE_THUMB_HEIGHT / matClone.height());
                Imgproc.resize(matClone, resized,
                    new Size(matClone.width() * scale, matClone.height() * scale));
                Imgcodecs.imwrite(thumbPath, resized);
                resized.release();
            } catch (Exception e) {
                System.err.println("Failed to save thumbnail: " + e.getMessage());
            } finally {
                matClone.release();
            }
        }
    }

    // Load thumbnail from cache directory
    @Override
    public boolean loadThumbnailFromCache(String cacheDir, int nodeIndex) {
        String thumbPath = cacheDir + File.separator + "node_" + nodeIndex + "_thumb.png";
        File thumbFile = new File(thumbPath);
        if (thumbFile.exists()) {
            try {
                Mat loaded = Imgcodecs.imread(thumbPath);
                if (!loaded.empty()) {
                    // Use setOutputMat which handles thread-safe assignment and thumbnail creation
                    setOutputMat(loaded.clone());

                    loaded.release();
                    return true;
                }
            } catch (Exception e) {
                System.err.println("Failed to load thumbnail: " + e.getMessage());
            }
        }
        return false;
    }

    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0; // Reset counter on start

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    // Take from input queue (blocks until available)
                    if (inputQueue == null) {
                        Thread.sleep(100);
                        continue;
                    }

                    Mat input = inputQueue.take();
                    incrementInputReads1(); // Track frames read from input
                    if (input == null) {
                        continue;
                    }

                    if (!enabled) {
                        // Bypass mode: move frame from input to output, update thumbnail
                        incrementWorkUnits();
                        setOutputMat(input.clone());
                        if (outputQueue != null) {
                            outputQueue.put(input);
                        } else {
                            input.release();
                        }
                        continue;
                    }

                    // Process the frame normally
                    Mat output = process(input);

                    // Increment work units regardless of output (even if null)
                    incrementWorkUnits();

                    // Update thumbnail and put on output queue
                    if (output != null) {
                        // Clone for persistent storage (outputMat for preview/thumbnail)
                        setOutputMat(output.clone());

                        // Clone for preview callback (callback may run async after output is released)
                        Mat previewClone = output.clone();
                        notifyFrame(previewClone);
                        // Note: previewClone will be released by the callback

                        // Check for backpressure BEFORE trying to put (so we can lower priority while blocked)
                        checkBackpressure();

                        if (outputQueue != null) {
                            // Clone for downstream node (they will release it)
                            outputQueue.put(output.clone());
                        }

                        // Release the original output from process()
                        output.release();
                    }

                    // Release input
                    input.release();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "Processing-" + name + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Serialize ProcessingNode-specific properties.
     * Subclasses should call super.serializeProperties() and then add their own.
     */
    @Override
    public void serializeProperties(JsonObject json) {
        // Only serialize enabled if false (default is true)
        if (!enabled) {
            json.addProperty("enabled", enabled);
        }
    }

    /**
     * Deserialize ProcessingNode-specific properties.
     * Subclasses should call super.deserializeProperties() and then read their own.
     */
    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("enabled")) {
            enabled = json.get("enabled").getAsBoolean();
        }
    }
}
