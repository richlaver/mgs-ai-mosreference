# MissionHelp Demo

The **MissionHelp Demo** is a Streamlit-based chatbot application designed to provide information about **MissionOS**, a platform for managing construction and instrumentation data. The app leverages **Retrieval-Augmented Generation (RAG)** to retrieve relevant text, images, and videos from a vector store and generate user-friendly responses. It integrates with **Google Cloud SQL** for image storage, **Qdrant** for vector search, and **Grok 3** (via xAI API) as the language model. The app is optimized for deployment on **Streamlit Community Cloud**.

## Features

- **Interactive Chatbot**: Users can ask questions about MissionOS, receiving responses with text, images, and YouTube videos sourced from the MissionOS manual.
- **RAG Pipeline**: Uses a LangGraph workflow to retrieve context from a Qdrant vector store and generate responses with the Grok 3 LLM.
- **Web Scraping**: Scrapes MissionOS manual webpages to extract content, images, and videos, storing images in a PostgreSQL database.
- **Admin Controls**: Allows admins to update database parameters (chunk size, overlap), reconfigure RAG settings (retrieval k), and evaluate retrieval performance (disabled until test set is available).
- **Vector Store Export**: Includes a standalone script (`export_vectors.py`) to export Qdrant vector store points to a CSV for analysis.

## Repository Structure

```
├── app.py                   # Main Streamlit app entry point
├── classes.py               # State classes for LangGraph workflow
├── database.py              # PostgreSQL operations and web scraping
├── evaluate_retrieval.py    # Retrieval accuracy evaluation (used by admin UI)
├── export_vectors.py        # Standalone script to export Qdrant points to CSV
├── rag.py                   # RAG pipeline using LangGraph
├── session.py               # Streamlit session state management
├── setup.py                 # Dependency and configuration setup
├── ui.py                    # Chatbot UI rendering
├── .gitignore               # Git ignore rules
├── .gitattributes           # Git attribute configurations
├── requirements.txt         # Python dependencies
├── packages.txt             # System packages
├── WUM articles.csv         # Webpage IDs for MissionOS manual scraping
├── .streamlit/
│   └── secrets.toml         # Streamlit secrets configuration
├── scrape_cache/            # Cache directory for scraped documents
```

## Prerequisites

To run or deploy the MissionHelp Demo, ensure the following:

- **Python 3.9+**
- **Streamlit Community Cloud** account for deployment
- **Google Cloud credentials** (`google_credentials.json`) for PostgreSQL access
- **xAI API key** for Grok 3 LLM
- **Qdrant credentials** for vector store access
- **Playwright browsers** for web scraping (installed automatically)
- A **secrets.toml** file in `.streamlit/` with:
  - `xai_api_key`: xAI API key
  - `GOOGLE_CREDENTIALS_JSON`: Google Cloud credentials JSON
  - `google_project_id`: Google Cloud project ID
  - `qdrant_client_credentials`: Qdrant connection details
  - `admin_password`: Password for admin UI access

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/missionhelp-demo.git
   cd missionhelp-demo
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install System Packages** (if needed):

   ```bash
   sudo apt-get install -y $(cat packages.txt)
   ```

4. **Set Up Secrets**: Create `.streamlit/secrets.toml` with the required credentials (see **Prerequisites**).

5. **Run Locally**:

   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Configure secrets in the Streamlit Cloud dashboard (same as `secrets.toml`).
4. Deploy the app, specifying `app.py` as the entry point.

Ensure `requirements.txt` and `packages.txt` are correctly set for dependency installation during deployment.

## Usage

### Chatbot Interface

- Access the app via the Streamlit URL or locally at `http://localhost:8501`.
- Enter questions about MissionOS in the chat input (e.g., "How do I manage instruments in MissionOS?").
- View responses with text, images, and embedded YouTube videos.
- The UI auto-scrolls to the latest message.

### Admin Controls

- Expand the "Admin log-in" section and enter the admin password.
- **Database Parameters**: Adjust `chunk_size` and `chunk_overlap`, then click "Update Database" to rebuild the vector store.
- **RAG Parameters**: Modify `retrieval_k` (number of chunks retrieved) and click "Reconfigure RAG".
- **Retrieval Evaluation**: Disabled until `retrieval_test_set.csv` is provided. Will display precision, recall, and MRR metrics.

### Exporting Vector Store

- Run `export_vectors.py` as a standalone script to export Qdrant points to `qdrant_points.csv`:

  ```bash
  python export_vectors.py
  ```
- The script is not integrated into the Streamlit UI.

## Notes

- **Retrieval Test**: The "Run Retrieval Test" button is disabled because `retrieval_test_set.csv` is not yet constructed. Create a CSV with columns `query`, `chunk_id`, `chunk_content`, and `source_url` to enable this feature.
- **Caching**: Web scraping results are cached in `scrape_cache/` to speed up database rebuilding. Set `use_cache=False` in `database.py` to force re-scraping.
- **Logging**: Logs are written to `scrape_debug.log`, `retrieval_debug.log`, and `qdrant_export.log` for debugging.
- **Image Storage**: Images are stored as binary data in a PostgreSQL database and referenced in responses via `[Image N]`.
- **Video Handling**: YouTube videos are embedded with titles extracted from webpage metadata.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or issues, open a GitHub issue or contact the maintainers.

---

Built with ❤️ by the Maxwell Geosystems AI team.