import argparse
import logging
from PIL import Image
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class FaceLabeler:
    def __init__(self, db_path="data/archive.db"):
        self.db_path = db_path
        # Mock database clusters load
        self.clusters = [
            {"id": 1, "faces": 12, "paths": ["data/faces/c1_1.jpg", "data/faces/c1_2.jpg"], "rep_crop": "data/faces/c1_rep.jpg"},
            {"id": 2, "faces": 5, "paths": ["data/faces/c2_1.jpg"], "rep_crop": "data/faces/c2_rep.jpg"}
        ]
        
    def start_labeling(self):
        print("Starting CLI Face Labeling Tool\n")
        print("Type 'skip' to skip a cluster, 'quit' to exit.")
        print("-" * 50)
        
        for cluster in self.clusters:
            print(f"Cluster ID: {cluster['id']}")
            print(f"Number of faces: {cluster['faces']}")
            print(f"Representative image paths: {', '.join(cluster['paths'])}")
            
            # Optionally show the crop
            self.show_image(cluster['rep_crop'])
            
            response = input(f"Enter name for Cluster {cluster['id']}: ").strip()
            
            if response.lower() == 'quit':
                print("Exiting tool.")
                break
            elif response.lower() == 'skip' or not response:
                print("Skipping...")
                continue
                
            self.save_identity(cluster['id'], response)
            print(f"Saved identity '{response}' for cluster {cluster['id']}.")
            print("-" * 50)
            
    def show_image(self, path):
        if os.path.exists(path):
            try:
                img = Image.open(path)
                img.show()
            except Exception as e:
                logger.error(f"Could not open image: {e}")
        else:
            logger.warning(f"Image {path} not found. Cannot display.")

    def save_identity(self, cluster_id, name):
        """Mock saving identity to database and linking faces."""
        logger.info(f"[DB Action] Created identity '{name}', linked to cluster {cluster_id}.")


def main():
    parser = argparse.ArgumentParser(description="CLI Face Labelling Tool")
    parser.add_argument("--db-path", type=str, default="data/archive.db", help="Path to database")
    
    args = parser.parse_args()
    
    labeler = FaceLabeler(db_path=args.db_path)
    labeler.start_labeling()

if __name__ == "__main__":
    main()
