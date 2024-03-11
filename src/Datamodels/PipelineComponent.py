import sqlalchemy
import psycopg2
from tools import load_json
import logging
from tools import get_filehandler
import json

class PipelineComponent:

    def __init__(self, configs, prefix=""):
        self.configs = configs
        self.postgres_credentials = load_json(prefix + f"/credentials/"
                                                       f"postgres/{self.configs['General']['run_name']}.json")
        self.prefix = prefix
        self.neo4j_credentials = load_json(self.prefix + "/credentials/neo4j.json")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        fh = get_filehandler(prefix, self.__class__.__name__)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.debug(f"Using configs {self.configs}")

    def get_extra_cursor(self):
        extra_conn = psycopg2.connect(host=self.postgres_credentials["host"], port=self.postgres_credentials["port"],
                                     database=self.postgres_credentials["database"],
                                     user=self.postgres_credentials["user"],
                                     password=self.postgres_credentials["password"])
        return extra_conn.cursor()

    def connect_databases(self):

        self.conn = psycopg2.connect(host=self.postgres_credentials["host"], port=self.postgres_credentials["port"],
                                     database=self.postgres_credentials["database"],
                                     user=self.postgres_credentials["user"],
                                     password=self.postgres_credentials["password"])
        # create sqlalchemy engine
        self.postgres_engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{self.postgres_credentials['user']}:{self.postgres_credentials['password']}@"
            f"{self.postgres_credentials['host']}:{self.postgres_credentials['port']}"
            f"/{self.postgres_credentials['database']}", execution_options=dict(stream_results=True))

        self.cur = self.conn.cursor()

        with self.conn.cursor(name='iter_cursor') as cursor:
            cursor.itersize = 100000  # chunk size

    def disconnect_databases(self):

        self.cur.close()
        self.conn.close()


    def database_exists(self, database_name):
        try:
            # Connect to the PostgreSQL server
            conn = psycopg2.connect(host=self.postgres_credentials["host"], port=self.postgres_credentials["port"],
                                    database="postgres",  # Connect to the default "postgres" database
                                    user=self.postgres_credentials["user"],
                                    password=self.postgres_credentials["password"])
            conn.autocommit = True  # Enable autocommit mode to execute SQL commands

            # Check if the database with the specified name already exists
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            exists = cursor.fetchone() is not None
            cursor.close()
            conn.close()


            return exists

        except Exception as e:
            self.logger.error(f"Error checking if PostgreSQL database exists: {str(e)}")
            return False

    def create_postgres_database(self):
        run_name = self.configs["General"]["run_name"]
        # get postgres credentials
        file = f"{self.prefix}/credentials/postgres/{run_name}.json"
        with open(file) as f:
            postgres_credentials = json.load(f)

        database_name = postgres_credentials["database"]

        if not self.database_exists(database_name):
            try:
                # Connect to the PostgreSQL server
                conn = psycopg2.connect(host=self.postgres_credentials["host"], port=self.postgres_credentials["port"],
                                        database="postgres",  # Connect to the default "postgres" database
                                        user=self.postgres_credentials["user"],
                                        password=self.postgres_credentials["password"])
                conn.autocommit = True  # Enable autocommit mode to execute database creation SQL

                # Create a new database with the specified name
                cursor = conn.cursor()
                cursor.execute(f"CREATE DATABASE {database_name}")
                cursor.close()
                conn.close()
                self.logger.info(f"PostgreSQL database '{database_name}' created successfully.")

            except Exception as e:
                self.logger.error(f"Error creating PostgreSQL database: {str(e)}")
        else:
            self.logger.info(f"PostgreSQL database '{database_name}' already exists.")


    def empty_database(self):
        try:

            self.connect_databases()
            self.conn.autocommit = True
            self.cur.execute("DROP SCHEMA public CASCADE")
            self.cur.execute("CREATE SCHEMA public")
            self.disconnect_databases()

            self.logger.info("Database deleted successfully.")

        except Exception as e:
            self.logger.error(f"Error deleting database: {str(e)}")