import sqlalchemy
from sqlalchemy import text, MetaData

from .db import DB
from google.cloud.sql.connector import Connector
from .util import (
    generate_ddl,
    get_db_secret,
    rate_limited_execute,
    DBResourceExhaustedError,
)
from typing import Any, Tuple
from threading import Semaphore

SCHEMA_QUERY = """
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, column_name;
"""


class PGDB(DB):

    def __init__(self, db_config):
        super().__init__(db_config)
        instance_connection_name = f"{db_config['project_id']}:{db_config['region']}:{db_config['instance_name']}"
        db_user = db_config["user_name"]
        db_pass_secret_path = db_config["password"]
        db_pass = get_db_secret(db_pass_secret_path)
        self.db_name = db_config["database_name"]
        self.db_config = db_config
        self.execs_per_minute = db_config["max_executions_per_minute"]
        self.semaphore = Semaphore(self.execs_per_minute)
        self.max_attempts = 3

        # Initialize the Cloud SQL Connector object
        connector = Connector()

        def getconn():
            conn = connector.connect(
                instance_connection_name,
                "pg8000",
                user=db_user,
                password=db_pass,
                db=self.db_name,
            )
            return conn

        self.engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_size=50,
            connect_args={"command_timeout": 60},
        )

    def get_metadata(self) -> dict:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, schema='public')

        db_metadata = {}
        for table in metadata.tables.values():
            columns = []
            for column in table.columns:
                columns.append({
                    'name': column.name,
                    'type': str(column.type)
                })
            db_metadata[table.name] = columns

        return db_metadata

    def generate_schema(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(SCHEMA_QUERY))
            headers = tuple(result.keys())
            rows = result.fetchall()
            return headers, rows

    def generate_ddl(self):
        headers, rows = self.generate_schema()
        return generate_ddl(rows, self.db_name)

    def _execute(self, query: str):
        result = []
        error = None
        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    resultset = connection.execute(text(query))
                    if resultset.returns_rows:
                        rows = resultset.fetchall()
                        result.extend(r._asdict() for r in rows)
        except Exception as e:
            error = str(e)
            if "57P03" in error:
                raise DBResourceExhaustedError("DB Exhausted") from e
        return result, error

    def create_database(self, database_name: str):
        result = []
        error = None
        query = f"CREATE DATABASE {database_name}"
        try:
            with self.engine.connect() as connection:
                connection.execution_options(isolation_level="AUTOCOMMIT").execute(text(query))
        except Exception as e:
            error = str(e)
        return result, error

    def drop_database(self, database_name: str):
        result = []
        error = None
        query = f"DROP DATABASE {database_name}"
        try:
            with self.engine.connect() as connection:
                connection.execution_options(isolation_level="AUTOCOMMIT").execute(text(query))
        except Exception as e:
            error = str(e)
        return result, error

    def execute_dml(self, query: str, eval_query: str = None):
        result = []
        eval_result = []
        error = None
        try:
            with self.engine.connect() as connection:
                with connection.begin() as transaction:
                    resultset = connection.execute(text(query))
                    if resultset.returns_rows:
                        rows = resultset.fetchall()
                        result.extend(r._asdict() for r in rows)

                    if eval_query:
                        eval_resultset = connection.execute(text(eval_query))
                        if eval_resultset.returns_rows:
                            eval_rows = eval_resultset.fetchall()
                            eval_result.extend(r._asdict() for r in eval_rows)

                    transaction.rollback()
        except Exception as e:
            error = str(e)
        return result, eval_result, error

    def execute(self, query: str) -> Tuple[Any, float]:
        if isinstance(self.execs_per_minute, int):
            return rate_limited_execute(
                query,
                self._execute,
                self.execs_per_minute,
                self.semaphore,
                self.max_attempts,
            )
        else:
            return self._execute(query)
