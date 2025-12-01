# How to Test Conversations with Your Model

You can test any conversation (like one with ChatGPT) using the emotion drift detection model. Here are the different ways to provide conversations:

## Quick Start

### Option 1: Simple Text Format (Easiest)

Create a text file with your conversation:

```
User: Hello, I need help with my account
Assistant: Sure, I can help. What do you need?
User: I'm having trouble logging in
Assistant: Let me help you with that
User: This is really frustrating, nothing is working!
Assistant: I understand your frustration. Let me escalate this.
User: Thank you, I appreciate it
```

Save it as `conversation.txt`, then run:
```bash
python test_conversation.py --file conversation.txt
```

### Option 2: Plain Text (One Message Per Line)

You can also just put one message per line:

```
Hello, I need help with my account
Sure, I can help. What do you need?
I'm having trouble logging in
Let me help you with that
This is really frustrating, nothing is working!
I understand your frustration. Let me escalate this.
Thank you, I appreciate it
```

The script will automatically detect this format.

### Option 3: JSON Format

For structured data, use JSON:

```json
[
  {"speaker": "User", "text": "Hello, I need help with my account"},
  {"speaker": "Assistant", "text": "Sure, I can help. What do you need?"},
  {"speaker": "User", "text": "I'm having trouble logging in"},
  {"speaker": "Assistant", "text": "Let me help you with that"},
  {"speaker": "User", "text": "This is really frustrating, nothing is working!"},
  {"speaker": "Assistant", "text": "I understand your frustration. Let me escalate this."},
  {"speaker": "User", "text": "Thank you, I appreciate it"}
]
```

Save as `conversation.json`, then run:
```bash
python test_conversation.py --file conversation.json
```

### Option 4: Command-Line Arguments

You can also pass messages directly:
```bash
python test_conversation.py --text "Hello, I need help" "Sure, what can I help with?" "I'm frustrated"
```

## Example: Testing a ChatGPT Conversation

### Step 1: Copy Your Conversation

If you had a conversation with ChatGPT, copy it in this format:

```
You: Can you help me understand Python classes?
ChatGPT: Of course! I'd be happy to help you understand Python classes. What specifically would you like to know?
You: I'm confused about inheritance
ChatGPT: No problem! Inheritance in Python allows a class to inherit attributes and methods from another class...
You: This is still confusing, can you give me a simpler example?
ChatGPT: Absolutely! Let me break it down with a very simple example...
You: Perfect, thank you! That makes much more sense now.
```

### Step 2: Save to File

Save it as `chatgpt_conversation.txt`

### Step 3: Run Analysis

```bash
cd src
python test_conversation.py --file chatgpt_conversation.txt --checkpoint models/bert_real_weighted/best_model.pt
```

### What You'll Get

The script will:
1. **Show emotion for each message**
   - Turn 1 [You]: Emotion: NEUTRAL (85% confidence)
   - Turn 2 [ChatGPT]: Emotion: JOY (72% confidence)
   - etc.

2. **Detect emotion drifts**
   - When emotions change between messages
   - Flag significant shifts (e.g., neutral to anger)

3. **Provide a summary**
   - Total messages
   - Number of emotion drifts
   - Emotion distribution across the conversation

## Example Output

```
======================================================================
CONVERSATION EMOTION ANALYSIS
======================================================================

Turn 1 [User]
  Text: I have a question about my account
  Emotion: NEUTRAL (92.3% confidence)

Turn 2 [Assistant]
  Text: Sure, I can help. What do you need?
  Emotion: JOY (78.5% confidence)

Turn 3 [User]
  Text: I'm having trouble logging in
  Emotion: NEUTRAL (84.2% confidence)

Turn 4 [Assistant]
  Text: Let me help you with that
  Emotion: NEUTRAL (76.1% confidence)

Turn 5 [User]
  Text: This is really frustrating, nothing is working!
  Emotion: ANGER (67.8% confidence)

[DRIFT #1] Turn 4 to Turn 5
  Assistant: NEUTRAL to User: ANGER
  Magnitude: 4 steps
  [WARNING] SIGNIFICANT DRIFT - Large emotion shift detected!

======================================================================
SUMMARY
======================================================================
Total messages: 5
Emotion drifts detected: 1
Significant drifts: 1

Emotion distribution:
  NEUTRAL: 3 (60.0%)
  JOY: 1 (20.0%)
  ANGER: 1 (20.0%)
```

## Tips

1. **Speaker labels are optional** - The script works even if you don't label speakers
2. **Format is flexible** - Mix different formats and the script will try to parse it
3. **Long messages work** - The model handles messages up to 128 tokens automatically
4. **Multiple conversations** - Create separate files for different conversations

## Common Use Cases

### Customer Support Conversation
```bash
python test_conversation.py --file support_chat.txt
```

### Interview/Feedback Session
```bash
python test_conversation.py --file interview.txt
```

### Therapy/Counseling Session
```bash
python test_conversation.py --file therapy_session.txt
```

### Educational Dialogue
```bash
python test_conversation.py --file tutoring_session.txt
```

## Advanced Options

### Use a Different Model
```bash
python test_conversation.py --file conversation.txt --checkpoint models/roberta_real/best_model.pt --model-name roberta-base
```

### Use CPU Instead of GPU
```bash
python test_conversation.py --file conversation.txt --device cpu
```

## Getting Help

If you run into issues:
1. Check that your checkpoint path is correct
2. Make sure the conversation file exists and is readable
3. Verify your model was trained successfully
4. Try with the example conversation first (run without --file argument)

## Example Conversation Files

I've created example conversation templates you can modify:
- Copy any conversation you have
- Format it as shown above
- Save as `.txt` or `.json`
- Run the analysis!

