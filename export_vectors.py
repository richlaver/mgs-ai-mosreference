"""Export Qdrant vector store points to a CSV file.

Scrolls through the 'manual-text' collection, extracting point IDs and payloads,
and writes them to a CSV for analysis of MissionOS chatbot chunks.
"""

import csv
import logging
import streamlit as st
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qdrant_export.log")
    ]
)
logger = logging.getLogger(__name__)

def export_points_to_csv(output_file: str = "qdrant_points.csv", limit: int = 1000) -> None:
    """Scroll through Qdrant points and write to CSV.

    Args:
        output_file: Path to output CSV file.
        limit: Maximum number of points to retrieve.
    """
    try:
        # Initialize Qdrant client
        client = QdrantClient(**st.secrets.qdrant_client_credentials)
        logger.info("Connected to Qdrant client")

        # Scroll through points
        points = client.scroll(collection_name="manual-text", limit=limit)[0]
        logger.info(f"Retrieved {len(points)} points from manual-text collection")

        # Define CSV columns
        fieldnames = ["point_id", "source", "page_content", "videos"]
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Write each point
            for i, point in enumerate(points):
                payload = point.payload or {}
                
                # Debug: Log payload keys for first few points
                if i < 5:
                    logger.debug(f"Point {point.id} payload keys: {list(payload.keys())}")
                    logger.debug(f"Point {point.id} payload: {payload}")

                # Extract metadata
                metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
                source = metadata.get("source", "")
                videos = str(metadata.get("videos", []))

                row = {
                    "point_id": str(point.id),
                    "source": source,
                    # "page_content": payload.get("page_content", "")[:200],  # Truncate for readability
                    "page_content": payload.get("page_content", ""),
                    "videos": videos
                }
                writer.writerow(row)
                logger.debug(f"Wrote point {point.id} to CSV")

        logger.info(f"Exported {len(points)} points to {output_file}")

    except Exception as e:
        logger.error(f"Error exporting points to CSV: {str(e)}")
        raise

def main():
    """Run the export process."""
    export_points_to_csv()

if __name__ == "__main__":
    main()