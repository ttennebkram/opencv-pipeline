package com.ttennebkram.pipeline.ui;

import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.ScrolledComposite;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;

import com.ttennebkram.pipeline.nodes.*;
import com.ttennebkram.pipeline.registry.NodeRegistry;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Shared toolbar builder for creating searchable node toolbars.
 * Used by both PipelineEditor and ContainerEditorWindow.
 */
public class ToolbarBuilder {

    private Display display;
    private Shell shell;
    private Composite toolbarContainer;
    private Text searchBox;
    private ScrolledComposite scrolledToolbar;
    private Composite toolbarContent;
    private List<SearchableButton> searchableButtons = new ArrayList<>();
    private int selectedButtonIndex = -1;

    // Callbacks for adding nodes
    private Runnable addFileSourceNode;
    private Runnable addWebcamSourceNode;
    private Runnable addBlankSourceNode;
    private java.util.function.Consumer<String> addNodeByType;

    /**
     * Inner class to track searchable toolbar buttons.
     */
    public static class SearchableButton {
        public Button button;
        public String nodeName;
        public String category;
        public Runnable action;
        public Label categoryLabel;
        public Label separator;
        public boolean isFirstInCategory;
    }

    public ToolbarBuilder(Display display, Shell shell) {
        this.display = display;
        this.shell = shell;
    }

    public void setAddFileSourceNode(Runnable callback) {
        this.addFileSourceNode = callback;
    }

    public void setAddWebcamSourceNode(Runnable callback) {
        this.addWebcamSourceNode = callback;
    }

    public void setAddBlankSourceNode(Runnable callback) {
        this.addBlankSourceNode = callback;
    }

    public void setAddNodeByType(java.util.function.Consumer<String> callback) {
        this.addNodeByType = callback;
    }

    /**
     * Build the toolbar in the given parent composite.
     */
    public Composite build(Composite parent) {
        // Create outer container for search box + scrollable toolbar
        toolbarContainer = new Composite(parent, SWT.BORDER);
        toolbarContainer.setLayoutData(new GridData(SWT.FILL, SWT.FILL, false, true));
        GridLayout containerLayout = new GridLayout(1, false);
        containerLayout.marginWidth = 0;
        containerLayout.marginHeight = 0;
        containerLayout.verticalSpacing = 0;
        toolbarContainer.setLayout(containerLayout);

        // Search label and box at top
        Label searchLabel = new Label(toolbarContainer, SWT.NONE);
        searchLabel.setText("Type to Search:");
        Font searchFont = new Font(display, "Arial", 13, SWT.BOLD);
        searchLabel.setFont(searchFont);
        searchLabel.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        searchBox = new Text(toolbarContainer, SWT.BORDER | SWT.SEARCH | SWT.ICON_SEARCH | SWT.ICON_CANCEL);
        searchBox.setMessage("Search nodes...");
        searchBox.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        searchBox.addModifyListener(e -> {
            clearButtonHighlighting();
            filterToolbarButtons();
        });
        searchBox.addListener(SWT.KeyDown, e -> {
            if (e.keyCode == SWT.ESC) {
                searchBox.setText("");
            }
        });
        searchBox.addListener(SWT.MouseDown, e -> {
            Rectangle bounds = searchBox.getBounds();
            if (e.x > bounds.width - 20) {
                searchBox.setText("");
            }
        });
        searchBox.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.keyCode == SWT.CR || e.keyCode == SWT.KEYPAD_CR) {
                    addSelectedNode();
                } else if (e.keyCode == SWT.ARROW_DOWN) {
                    navigateSelection(1);
                    e.doit = false;
                } else if (e.keyCode == SWT.ARROW_UP) {
                    navigateSelection(-1);
                    e.doit = false;
                }
            }
        });

        // Create scrollable container for the toolbar
        scrolledToolbar = new ScrolledComposite(toolbarContainer, SWT.V_SCROLL);
        scrolledToolbar.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        scrolledToolbar.setExpandHorizontal(true);
        scrolledToolbar.setExpandVertical(true);

        toolbarContent = new Composite(scrolledToolbar, SWT.NONE);
        GridLayout toolbarLayout = new GridLayout(1, false);
        toolbarLayout.verticalSpacing = 0;
        toolbarLayout.marginHeight = 5;
        toolbarLayout.marginWidth = 12;
        toolbarContent.setLayout(toolbarLayout);

        // Set green background
        Color toolbarGreen = new Color(160, 200, 160);
        toolbarContent.setBackground(toolbarGreen);
        scrolledToolbar.setBackground(toolbarGreen);
        toolbarContainer.setBackground(toolbarGreen);

        Font boldFont = new Font(display, "Arial", 13, SWT.BOLD);

        // Sources section
        Label sourcesLabel = new Label(toolbarContent, SWT.NONE);
        sourcesLabel.setText("Sources:");
        sourcesLabel.setFont(boldFont);
        sourcesLabel.setBackground(toolbarGreen);
        sourcesLabel.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        createSearchableButton("File Source", "Sources", addFileSourceNode, sourcesLabel, null, true);
        createSearchableButton("Webcam Source", "Sources", addWebcamSourceNode, null, null, false);
        createSearchableButton("Blank Source", "Sources", addBlankSourceNode, null, null, false);

        // Generate buttons from NodeRegistry grouped by category
        Map<String, List<String[]>> categoryNodes = new LinkedHashMap<>();

        for (NodeRegistry.NodeRegistration info : NodeRegistry.getAllNodes()) {
            // Skip source nodes - they have their own toolbar section
            if (SourceNode.class.isAssignableFrom(info.nodeClass)) {
                continue;
            }
            // Skip container boundary nodes - they're auto-created, not user-addable
            if (info.name.equals("ContainerInput") || info.name.equals("ContainerOutput")) {
                continue;
            }
            // Create temp node to get display name and category
            ProcessingNode tempNode = NodeRegistry.createProcessingNode(info.name, display, shell, 0, 0);
            if (tempNode != null) {
                String displayName = tempNode.getDisplayName();
                String category = tempNode.getCategory();
                tempNode.disposeThumbnail();

                categoryNodes.computeIfAbsent(category, k -> new ArrayList<>())
                    .add(new String[]{displayName, info.name});
            }
        }

        // Create buttons for each category (sorted by category name, then by node name within each category)
        List<String> sortedCategories = new ArrayList<>(categoryNodes.keySet());
        java.util.Collections.sort(sortedCategories);

        for (String category : sortedCategories) {
            List<String[]> nodeList = categoryNodes.get(category);
            // Sort nodes alphabetically within each category
            nodeList.sort((a, b) -> a[0].compareToIgnoreCase(b[0]));

            // Separator
            Label separator = new Label(toolbarContent, SWT.SEPARATOR | SWT.HORIZONTAL);
            separator.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

            // Category label
            Label categoryLabel = new Label(toolbarContent, SWT.NONE);
            categoryLabel.setText(category + ":");
            categoryLabel.setFont(boldFont);
            categoryLabel.setBackground(toolbarGreen);
            categoryLabel.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

            // Buttons for this category
            boolean isFirst = true;
            for (String[] nodeInfo : nodeList) {
                String displayName = nodeInfo[0];
                String registryName = nodeInfo[1];
                createSearchableButton(displayName, category,
                    () -> { if (addNodeByType != null) addNodeByType.accept(registryName); },
                    isFirst ? categoryLabel : null,
                    isFirst ? separator : null,
                    isFirst);
                isFirst = false;
            }
        }

        scrolledToolbar.setContent(toolbarContent);
        scrolledToolbar.setMinSize(toolbarContent.computeSize(SWT.DEFAULT, SWT.DEFAULT));

        return toolbarContainer;
    }

    private void createSearchableButton(String text, String category, Runnable action,
            Label categoryLabel, Label separator, boolean isFirstInCategory) {
        Button btn = new Button(toolbarContent, SWT.PUSH | SWT.FLAT);
        btn.setText(text);
        btn.setBackground(new Color(160, 160, 160));
        GridData gd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        gd.heightHint = btn.computeSize(SWT.DEFAULT, SWT.DEFAULT).y + 2;
        btn.setLayoutData(gd);
        btn.addListener(SWT.Selection, e -> { if (action != null) action.run(); });

        SearchableButton sb = new SearchableButton();
        sb.button = btn;
        sb.nodeName = text;
        sb.category = category;
        sb.action = action;
        sb.categoryLabel = categoryLabel;
        sb.separator = separator;
        sb.isFirstInCategory = isFirstInCategory;
        searchableButtons.add(sb);
    }

    private void filterToolbarButtons() {
        String searchText = searchBox.getText().trim().toLowerCase();

        Map<String, Boolean> categoryVisible = new HashMap<>();

        for (SearchableButton sb : searchableButtons) {
            boolean visible;
            if (searchText.isEmpty()) {
                visible = true;
            } else {
                String[] searchWords = searchText.split("[\\s/\\-_.,]+");
                String searchableText = (sb.nodeName + " " + sb.category).toLowerCase();

                visible = true;
                for (String searchWord : searchWords) {
                    if (searchWord.isEmpty()) continue;
                    if (!searchableText.contains(searchWord)) {
                        visible = false;
                        break;
                    }
                }
            }

            sb.button.setVisible(visible);
            ((GridData) sb.button.getLayoutData()).exclude = !visible;

            if (visible) {
                categoryVisible.put(sb.category, true);
            } else if (!categoryVisible.containsKey(sb.category)) {
                categoryVisible.put(sb.category, false);
            }
        }

        // Update category label and separator visibility
        for (SearchableButton sb : searchableButtons) {
            if (sb.isFirstInCategory && sb.categoryLabel != null) {
                boolean catVisible = categoryVisible.getOrDefault(sb.category, false);
                sb.categoryLabel.setVisible(catVisible);
                ((GridData) sb.categoryLabel.getLayoutData()).exclude = !catVisible;
                if (sb.separator != null) {
                    sb.separator.setVisible(catVisible);
                    ((GridData) sb.separator.getLayoutData()).exclude = !catVisible;
                }
            }
        }

        toolbarContent.layout(true);
        scrolledToolbar.setMinSize(toolbarContent.computeSize(SWT.DEFAULT, SWT.DEFAULT));
    }

    private void clearButtonHighlighting() {
        for (SearchableButton sb : searchableButtons) {
            sb.button.setBackground(new Color(160, 160, 160));
        }
        selectedButtonIndex = -1;
    }

    private void navigateSelection(int delta) {
        List<Integer> visibleIndices = new ArrayList<>();
        for (int i = 0; i < searchableButtons.size(); i++) {
            if (searchableButtons.get(i).button.isVisible()) {
                visibleIndices.add(i);
            }
        }
        if (visibleIndices.isEmpty()) return;

        clearButtonHighlighting();

        int currentPos = visibleIndices.indexOf(selectedButtonIndex);
        int newPos;
        if (currentPos < 0) {
            newPos = delta > 0 ? 0 : visibleIndices.size() - 1;
        } else {
            newPos = currentPos + delta;
            if (newPos < 0) newPos = visibleIndices.size() - 1;
            if (newPos >= visibleIndices.size()) newPos = 0;
        }

        selectedButtonIndex = visibleIndices.get(newPos);
        SearchableButton selected = searchableButtons.get(selectedButtonIndex);
        selected.button.setBackground(new Color(100, 150, 255));
    }

    private void addSelectedNode() {
        if (selectedButtonIndex >= 0 && selectedButtonIndex < searchableButtons.size()) {
            SearchableButton sb = searchableButtons.get(selectedButtonIndex);
            if (sb.button.isVisible() && sb.action != null) {
                sb.action.run();
                return;
            }
        }
        for (SearchableButton sb : searchableButtons) {
            if (sb.button.isVisible() && sb.action != null) {
                sb.action.run();
                break;
            }
        }
    }

    public Text getSearchBox() {
        return searchBox;
    }

    public List<SearchableButton> getSearchableButtons() {
        return searchableButtons;
    }
}
