import os.path
import sqlite3
from dataclasses import dataclass
import logging
logger = logging.getLogger("ehri_graph_rag")

@dataclass
class ActivityRecord:
    prompt: str
    mode: str
    model: str
    output: str
    error: str

class DatabaseManager:

    def __init__(self):
        self.db = "db/activity.db"
        self.connection = None
        self._create()

    def _connect(self):
        self.connection = sqlite3.connect(self.db)

    def _create(self):
        if not os.path.isfile(self.db):
            logger.info("Creating database for registering activity")
            self._connect()
            with open("db/creation.sql", "r") as file:
                logger.info("Creating table activity in the database")
                self.connection.executescript(file.read())
            self.connection.close()

    def insert_activity(self, activity: ActivityRecord):
        self._connect()
        data = [(activity.prompt, activity.mode, activity.model, activity.output, activity.error if activity.error else "")]
        cursor = self.connection.cursor()
        try:
            cursor.executemany("""INSERT INTO activity (timestamp, prompt, mode, model, output, error)
            VALUES(CURRENT_TIMESTAMP, ?, ?, ?, ?, ?);""""", data)
            self.connection.commit()
        except sqlite3.Error as e:
            logger.error("Error while inserting data into the database", e)
        finally:
            cursor.close()
            self.connection.close()