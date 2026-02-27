# Topic 6: Vision-Language Models (VLM)

Team: Shaina Kumar and Janice Guo 

This directory contains our implementations for Topic 6 using **Ollama + LLaVA** to build:

1. A multi-turn Vision-Language chat agent (LangGraph + Gradio)
2. A video surveillance pipeline that detects entry/exit events via frame sampling

---

# Table of Contents 

```
Topic6VLM/
├── README.md
├── 2minute.mov
├── frames/
├── outputs/
├── langgraph_vision_chat.py
├── video_surveillance.py
├── modified_surveillance.py
└── mod2_surveillance.py
```

## Files

- `2minute.mov`  
  Two-minute test video recorded in Thornton Hall.

- `frames/`  
  Extracted frames sampled approximately every 2 seconds.

- `outputs/`
  - `Lab6Exercise1.jpg` — Screenshot of Exercise 1 Gradio interface and sample interaction.
  - `llava7bmistral.txt` — Exercise 2 run using 7B Mistral quantized model.
  - `llava13b.txt` — Exercise 2 run using 13B model.
  - `llava13_resized.txt` — 13B run with resized frames.

- `langgraph_vision_chat.py`  
  Exercise 1 implementation.

- `video_surveillance.py`  
  Initial surveillance pipeline.

- `modified_surveillance.py`, `mod2_surveillance.py`  
  Surveillance variants with 13B model and resizing.

--- 
## Exercise 1 — Vision-Language LangGraph Chat Agent

This exercise implements a multi-turn vision-language chat agent that allows a user to upload an image and ask questions about it across multiple conversational turns. The system is built using LangGraph for structured state management and Gradio for the web-based user interface. The way it works is that the user uploads an image in the Gradio interface and clicks “Load Image,” which stores the image path inside LangGraph state through an initial `graph.invoke()` call. When the user submits a question, each submission triggers exactly one `graph.invoke()` execution. LangGraph routes the input through two nodes. The first node, `ingest_user_turn`, validates the input, handles special commands such as `verbose`, `quiet`, or `quit`, and appends a `HumanMessage` to conversation history. The second node, `call_llava`, constructs the Ollama message list from a rolling conversation window, resizes the image if necessary, calls the LLaVA model, and appends the resulting `AIMessage` reply. A `SqliteSaver` checkpoint mechanism stores the full conversation state to a SQLite database between HTTP calls so that history persists across turns. A rolling context window of 6 is used to limit the number of past messages sent to the model, preventing unbounded prompt growth. 

For this exercise, we used `llava:7b-v1.6-mistral-q4_0`, as we found it to be the more stable/less prone to crashing. From `outputs/Lab6Exercise1.jpg`, when asked “How many people are in this image?”, the model responded that there were four people. In reality, there are three main women in the foreground. The model likely counted background faces or possibly misinterpreted large background elements such as the balloon. When asked “Are there boys or girls in the image?”, the model correctly inferred the graduation context by identifying caps, gowns, and decorations. It correctly recognized that the individuals were young women. However, it incorrectly stated that there was “one girl and three young women,” again overcounting because all three people in the foreground are young women. Therefore, the model performs well at contextual inference and could reason about what is happening in the image. However, it struggles with precise counting and distinguishing foreground and background entities. 

--- 
## Exercise 2 — Video Surveillance Agent

This exercise implements a simple video surveillance system using a vision-language model. The video is decomposed into individual frames and each frame is analyzed independently. The goal is to detect when a person enters and exits the scene by monitoring changes in model predictions over time. Our pipeline begins by loading a two-minute video we recorded in Thornton Hall. Using OpenCV, the video is sampled approximately once every two seconds. Each sampled frame is resized to a maximum side length of 336 pixels to reduce memory usage and inference time, and then saved to the `frames/` directory. For each frame, the system sends the image to LLaVA with the prompt: “Is there a person visible in this scene? Reply with only one word: YES or NO.” The model’s response is parsed and converted into a boolean value representing whether a person is present. The system then tracks state transitions across frames. A transition from NO to YES is interpreted as an ENTER event, while a transition from YES to NO is interpreted as an EXIT event. After processing all frames, the system prints a timestamped report summarizing detected entry and exit times, along with the duration of each detected presence. The test video was recorded in Thornton Hall and is slightly shaky because my partner Janice and I swapped who was holding the phone during recording. The known ground truth events are as follows. I enter the frame around 00:02 and leave by approximately 00:05. Two random students walk by between approximately 00:54 and 00:58. Janice enters the frame around 01:13 and leaves around 01:18. These timestamps provide a reference for evaluating the accuracy of the model’s detections. Across multiple runs using different variants (`llava:7b-v1.6-mistral-q4_0` and 13B versions (max side lengths 336 and 512)), the system did seem to consistently detect these key appearances in the video. However, the system also produced several short false positives. We assume that the likely causes might include camera shake or motion blur, but it could just be random hallucinations. Also, given that the frames are analyzed independently without temporal smoothing, a single misclassified frame can generate a short visit in the report.

Link to Google Colab: https://colab.research.google.com/drive/13QqM8AqbuS3qI9TJDXFgIUiEg4BZ7G1l?usp=sharing 

