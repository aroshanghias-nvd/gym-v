"""MiniWoB task configurations.

This file contains all 125 MiniWoB++ tasks from BrowserGym.
Each task is configured with:
  - max_steps: Maximum episode steps
  - actions: Action subsets (bid=element ID, coord=coordinates)
  - description: Task description

Auto-generated from BrowserGym miniwob/all.py
"""

MINIWOB_TASKS = {
    "ascending-numbers": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click on the numbers in ascending order.",
    },
    "bisect-angle": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Find the line that bisects an angle evenly in two.",
    },
    "book-flight": {
        "max_steps": 20,
        "actions": ["bid"],
        "description": "Search for flight results.",
    },
    "book-flight-nodelay": {
        "max_steps": 20,
        "actions": ["bid"],
        "description": "[book-flight] Removed animation.",
    },
    "buy-ticket": {
        "max_steps": 20,
        "actions": ["bid"],
        "description": "Buy a ticket that matches the requested criteria.",
    },
    "choose-date": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Learn to operate a date picker tool.",
    },
    "choose-date-easy": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[choose-date] December only.",
    },
    "choose-date-medium": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[choose-date] December or November only.",
    },
    "choose-date-nodelay": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[choose-date] Removed animation.",
    },
    "choose-list": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Choose an item from a drop down list.",
    },
    "circle-center": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Find the center of a circle.",
    },
    "click-button": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on a specific button in a generated form.",
    },
    "click-button-sequence": {
        "max_steps": 30,
        "actions": ["bid", "coord"],
        "description": "Click on buttons in a certain order.",
    },
    "click-checkboxes": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click desired checkboxes.",
    },
    "click-checkboxes-large": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[click-checkboxes] Click at least 5 out of up to 12 checkboxes.",
    },
    "click-checkboxes-soft": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[click-checkboxes] Paraphrased entries.",
    },
    "click-checkboxes-transfer": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[click-checkboxes] Train and test on different number of targets.",
    },
    "click-collapsible": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click a collapsible element to expand it.",
    },
    "click-collapsible-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find and click on a specified link, from collapsible elements.",
    },
    "click-collapsible-2-nodelay": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[click-collapsible-2] Removed animation.",
    },
    "click-collapsible-nodelay": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[click-collapsible] Removed animation.",
    },
    "click-color": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click the specified color.",
    },
    "click-dialog": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click the button to close the dialog box.",
    },
    "click-dialog-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click a specific button in a dialog box.",
    },
    "click-link": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on a specified link in text.",
    },
    "click-menu": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click menu items.",
    },
    "click-menu-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find a specific item from a menu.",
    },
    "click-option": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click option boxes.",
    },
    "click-pie": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click items on a pie menu.",
    },
    "click-pie-nodelay": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "[click-pie] Removed animation.",
    },
    "click-scroll-list": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click multiple items from a scroll list.",
    },
    "click-shades": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click the shades that match a specified color.",
    },
    "click-shape": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on a specific shape.",
    },
    "click-tab": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on a tab element.",
    },
    "click-tab-2": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click a link inside a specific tab element.",
    },
    "click-tab-2-easy": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "[click-tab-2] One 1 tab.",
    },
    "click-tab-2-hard": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "[click-tab-2] Varying number of tabs from 2 to 6.",
    },
    "click-tab-2-medium": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "[click-tab-2] Choose between a link or ‘no match’.",
    },
    "click-test": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on a single button.",
    },
    "click-test-2": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on one of two buttons.",
    },
    "click-test-transfer": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "[click-test] Different buttons during train and test.",
    },
    "click-widget": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Click on a specific widget in a generated form.",
    },
    "copy-paste": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Copy text and paste it into an input.",
    },
    "copy-paste-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Copy text from a specific textarea and paste it into an input.",
    },
    "count-shape": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Count number of shapes.",
    },
    "count-sides": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Count the number of sides on a shape.",
    },
    "daily-calendar": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Create an event on a daily calendar.",
    },
    "drag-box": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag the smaller box into the larger box.",
    },
    "drag-circle": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag an item in a specified direction.",
    },
    "drag-cube": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag a 3D cube to show a specific face.",
    },
    "drag-items": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag items in a list, in a specified direction",
    },
    "drag-items-grid": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag items in a 2D grid around.",
    },
    "drag-shapes": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag shapes into a box.",
    },
    "drag-shapes-2": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag shapes into boxes, categorized by type.",
    },
    "drag-single-shape": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag a randomly generated shape in a specified direction.",
    },
    "drag-sort-numbers": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Drag numbers into sorted ascending order.",
    },
    "draw-circle": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Draw a circle around a marked point.",
    },
    "draw-line": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Draw a line through a marked point.",
    },
    "email-inbox": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "Navigate through an email inbox and perform some actions.",
    },
    "email-inbox-delete": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] No scrolling + 1 subtask.",
    },
    "email-inbox-forward": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] No scrolling + 1 subtask.",
    },
    "email-inbox-forward-nl": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox-forward] varied instruction texts (30 templates).",
    },
    "email-inbox-forward-nl-turk": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox-forward] varied instruction texts (100 templates).",
    },
    "email-inbox-important": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] No scrolling + 1 subtask.",
    },
    "email-inbox-nl-turk": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] varied instruction texts (100 templates for each subtask).",
    },
    "email-inbox-noscroll": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] No scrolling.",
    },
    "email-inbox-reply": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] No scrolling + 1 subtask.",
    },
    "email-inbox-star-reply": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[email-inbox] No scrolling + 2 subtasks.",
    },
    "enter-date": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use the date input to pick the correct date.",
    },
    "enter-password": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Enter the password into the form.",
    },
    "enter-text": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Enter given text to a textfield.",
    },
    "enter-text-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Convert given text to upper or lower case.",
    },
    "enter-text-dynamic": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Enter dynamically generated text to a textfield.",
    },
    "enter-time": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Enter the specified time into the input.",
    },
    "find-greatest": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find the card with the greatest number.",
    },
    "find-midpoint": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find the shortest mid-point of two points.",
    },
    "find-word": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find nth word in a block of text.",
    },
    "focus-text": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Focus into a text input.",
    },
    "focus-text-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Focus on a specific text input.",
    },
    "form-sequence": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "Perform a series of instructions on a form.",
    },
    "form-sequence-2": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "Perform a series of instructions on a form.",
    },
    "form-sequence-3": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "Perform a series of instructions on a form.",
    },
    "generate-number": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Generate a random number that meets certain criteria.",
    },
    "grid-coordinate": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find the Cartesian coordinates on a grid.",
    },
    "guess-number": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Guess the number.",
    },
    "highlight-text": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Highlight all the text.",
    },
    "highlight-text-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Highlight the specified paragraph.",
    },
    "hot-cold": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find and click on the hot area.",
    },
    "identify-shape": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Identify a randomly generated shape.",
    },
    "login-user": {
        "max_steps": 20,
        "actions": ["bid"],
        "description": "Enter user login details into the form.",
    },
    "login-user-popup": {
        "max_steps": 20,
        "actions": ["bid"],
        "description": "[login-user] Random popup.",
    },
    "multi-layouts": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Fill in forms of varying layouts.",
    },
    "multi-orderings": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Fill in forms with shuffled field orderings.",
    },
    "navigate-tree": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "Navigate a file tree to find a specified file or folder.",
    },
    "number-checkboxes": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Draw a given number using checkboxes.",
    },
    "odd-or-even": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Mark each number as odd or even.",
    },
    "order-food": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Order food items from a menu.",
    },
    "phone-book": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Find a contact in a phone book.",
    },
    "read-table": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Read information out from a table.",
    },
    "read-table-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Read multiple pieces of information out from a table.",
    },
    "resize-textarea": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Resize a textarea in a given direction.",
    },
    "right-angle": {
        "max_steps": 10,
        "actions": ["bid", "coord"],
        "description": "Given two points, add a third point to create a right angle.",
    },
    "scroll-text": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Scroll through a text area element and enter last word into text area.",
    },
    "scroll-text-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Scroll through a text area in a given direction.",
    },
    "search-engine": {
        "max_steps": 20,
        "actions": ["bid"],
        "description": "Search through a bunch of results to find a specified link.",
    },
    "sign-agreement": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Sign a user agreement.",
    },
    "simple-algebra": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Solve for X.",
    },
    "simple-arithmetic": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Perform some arithmetic math operations.",
    },
    "social-media": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "Interact with a social media feed.",
    },
    "social-media-all": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[social-media] Do some action on all matching entries.",
    },
    "social-media-some": {
        "max_steps": 30,
        "actions": ["bid"],
        "description": "[social-media] Do some action on some matching entries.",
    },
    "stock-market": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Buy from the stock market below a specified price.",
    },
    "terminal": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use the terminal to delete a file.",
    },
    "text-editor": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Modify a text's style in a text-editor.",
    },
    "text-transform": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Enter slightly transformed text into a text box.",
    },
    "tic-tac-toe": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Win a game of tic-tac-toe.",
    },
    "unicode-test": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Click on the button with the correct Unicode text.",
    },
    "use-autocomplete": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use autocomplete element efficiently.",
    },
    "use-autocomplete-nodelay": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "[use-autocomplete] Removed delay.",
    },
    "use-colorwheel": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use a color wheel.",
    },
    "use-colorwheel-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use a color wheel given specific random color.",
    },
    "use-slider": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use a slider to select a particular value.",
    },
    "use-slider-2": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use sliders to create a given combination.",
    },
    "use-spinner": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Use a spinner to select given number.",
    },
    "visual-addition": {
        "max_steps": 10,
        "actions": ["bid"],
        "description": "Count the total number of blocks.",
    },
}
