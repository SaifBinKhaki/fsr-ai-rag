To take your local n8n pipeline into a production-grade web application with MLOps, you need to transition from "local execution" to "service-based architecture."

Since you have a functional n8n pipeline, your "MLOps" focus should be on **Observability, Versioning, and API-first exposure.** Here is your roadmap to production.

---

## 1. Deploying the Pipeline (Production Infrastructure)

To move off your laptop, you need a hosting environment for n8n and your vector database.

- **n8n Hosting:**
- **Self-Hosted:** Use **Railway** or **Render**. They provide one-click deployments for n8n using Docker, ensuring your workflow stays online 24/7.
- **n8n Cloud:** If your university/project has a budget, n8n Cloud is the "zero-ops" path, allowing you to focus purely on the logic.

- **Vector Database (Qdrant):**
- Stop using the `./qdrant_local_data` file path. Deploy a free cluster on **Qdrant Cloud** (1GB is plenty for ~25k vectors).
- Update your Python ingestion script and n8n credentials to point to the new `QDRANT_URL` and `QDRANT_API_KEY`.

---

## 2. Exposing Your Pipeline to a Web App

Do not try to "embed" n8n into your website's source code. Treat n8n as an **API Backend**.

1. **Change Trigger:** Replace your "Chat Trigger" node with the **Webhook** node in n8n.
2. **API Schema:** Set the Webhook to `POST`. When your web application sends a JSON request `{ "message": "How do I apply?" }`, n8n will process it through your RAG pipeline and return a response.
3. **Frontend:** Build a simple **Next.js** chat interface. Your frontend sends a fetch request to the n8n Webhook URL and displays the resulting `output_text` in a chat bubble.

---

## 3. Implementing "MLOps" (The Professional Touch)

This is what differentiates a "student project" from an "AI Engineering portfolio item."

### A. Observability (Logging)

In n8n, add a node after your Synthesis step to log every interaction to a **PostgreSQL** or **Supabase** table.

- **Log:** `timestamp`, `user_query`, `final_response`, `latency_ms`, and `route_taken` (Internal RAG vs. Web Fallback).
- **Why:** If a user complains about a wrong answer, you can look at the database and see the exact trace of the LLM’s reasoning.

### B. Evaluation (The "Eval" Harness)

MLOps means "Evaluation." Create a separate n8n workflow for **Automated Testing**.

- **Golden Dataset:** Create a JSON list of 20 "tough" questions and the "perfect" answers.
- **Regression Test:** Every time you change your system prompt, run your dataset through this workflow. If the new accuracy score is lower than the old one, you know you've broken something.

### C. Versioning

- **Git for n8n:** Treat your workflow JSON files like code. Use the **n8n Git integration** to push your workflow versions to a GitHub repository.
- **Semantic Versioning:** Tag your workflows in GitHub (e.g., `v1.0.0-stable`, `v1.1.0-experimental`). If your production pipeline breaks, you can instantly revert to the previous version.

---

## 4. Your MLOps Architecture Diagram (for README)

In your portfolio README, add an "MLOps Lifecycle" section:

> "Our MLOps lifecycle ensures reliability through a three-tier loop:
>
> 1. **Ingestion (ETL):** Automated Python-based scraper + Qdrant sync.
> 2. **Orchestration (n8n):** Version-controlled pipelines deployed via Docker on Railway.
> 3. **Monitoring:** Structured event logging into Supabase, enabling post-hoc analysis of retrieval confidence and latency."

---

**Sources:**

1. _Crawl4AI Documentation: Headless browser automation for LLM ingestion._
2. _Qdrant & Sentence-Transformers Documentation: Vector database ingestion and local embedding model workflows._
3. _n8n Documentation: Workflow automation and Git version control integration._
