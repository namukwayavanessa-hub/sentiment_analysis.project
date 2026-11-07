# Group NLP Project

Members: **Jimia** and **Vanessa**

This project has two apps: a sentiment analyser built by fine-tuning DistilBERT, and a chatbot that answers questions about a poultry farming guide using only that document.

---

## App 1: Sentiment analyser

- **Model:** https://huggingface.co/jimiakiden/Vanessa-and-Jimia-sentiment-model
- **Base model:** DistilBERT, fine-tuned on `stanfordnlp/imdb` (movie reviews)
- **Screenshots:** `screenshot1.png` (Gradio demo), `screenshot2.png` (API test output)
- **Test accuracy:** 0.__ **Test F1:** 0.__
- **Files:** `app.py` (Gradio demo), `main.py` (FastAPI), `requirements.txt`

### How to run the API

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Example

Send a POST request to `/predict`:

```json
{"text": "I love this"}
```

Response:

```json
{"label": "POSITIVE", "score": 0.99}
```

Empty text is rejected with a `400` error ("Text cannot be empty").

---

## App 2: Document chatbot

- **Chatbot name:** Poultry Farming Assistant
- **Document used:** Poultry farming guide uganda (10 pages)
- **Screenshots:** `screenshot3.png` (answer with page number), `screenshot4.png` ("I don't know" for an out-of-document question)
- **File:** `chatbot_app.py`

### How it works

1. The PDF text is extracted page by page with `PdfReader`.
2. The text is split into chunks, and each chunk is turned into an embedding (384 numbers) with Sentence Transformers.
3. When a user asks a question, the most similar chunks are retrieved, along with their page numbers.
4. The retrieved chunks and the question are sent to an LLM through Hugging Face Inference Providers. The system prompt tells it to answer only from the context and cite pages.
5. Gradio's `ChatInterface` provides the chat window. The last 6 messages are kept so follow-up questions work.

### Test results

| # | Question | Right page found? | Answer correct? | Invented anything? |
|---|----------|-------------------|-----------------|--------------------|
| 1 | What are the benefits of poultry farming? | Yes  | Yes | No |
| 2 |  What are the most Challenges to consider when planning to start a poultry farm? | Yes  | Yes| No |
| 3 | Steps when planning a Poultry Enterprise | Yes | Yes |  No |
| 4 | Who won the 2010 World Cup? | n/a | Said "not in the document"? | No |
| 5 | What is transport? | n/a | Said "not in the document"? |  No |

### What broke and how we fixed it

- **Gradio version error.** `gr.ChatInterface(..., type="messages")` raised `TypeError: unexpected keyword argument 'type'`. Newer Gradio versions use the messages format by default, so we removed the argument.
- **Follow-up questions retrieved poorly.** Searching only on the latest message (e.g. "what about for chicks?") found irrelevant chunks. We now include the previous user question in the search query.
- **Extra keys in chat history.** We keep only `role` and `content` before sending history to the LLM, because other keys can be rejected.
- **The bot answered from general knowledge.** We tightened `SYSTEM_PROMPT` so it answers only from the retrieved text, says when it cannot find the answer, and ignores instructions found inside the document text.
- **Our "try to break it" tests (5):**
  1. Off-topic question: when did Uganda gain its Independence?
  2. Rude message: shut up!
  3. "Ignore your instructions": i want to eat
  4. Trick question: If chickens have enough food but are not growing properly, does that automatically mean they need more feed? Explain
  5. ____: ____

---

## Our model vs the LLM

(Write your answer to the discussion question under the Compare cell in Section B1. You could cover: speed, cost, accuracy on your 5 sentences, what each one does well or badly, and which you would use in a real product and why.)

---

## Limitations

- The sentiment model was trained only on positive and negative English movie reviews. It may be wrong on neutral reviews, other domains, Luganda or mixed-language text.
- The chatbot only knows what is in one PDF. If the PDF is wrong or outdated, so are the answers.
- The chatbot may occasionally miss the right page or answer imperfectly, so important decisions (vaccination schedules, medication, sick birds) should be checked with a veterinarian or the original document.
- Scanned PDFs (pictures of pages) do not work because there is no text to extract.
- We did not deploy to Hugging Face Spaces because Spaces hosting is now paid. Both apps run inside Colab.

---

## Who did what

- **Member A (FIRSTNAME):** ____
- **Member B (FIRSTNAME):** ____
