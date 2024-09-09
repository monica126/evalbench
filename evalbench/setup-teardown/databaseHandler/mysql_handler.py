import os
import csv
from typing import Any, Tuple, List
from .db_handler import DBHandler
from databases import get_database

drop_all_tables_query = [
        "SET GROUP_CONCAT_MAX_LEN = 32768;",
        "SET @tables = NULL;",
        "SELECT GROUP_CONCAT('`', table_name, '`') INTO @tables FROM information_schema.tables \
        WHERE table_schema = (SELECT DATABASE());",
        "SET @tables = CONCAT('DROP TABLE IF EXISTS ', @tables);",
        "PREPARE stmt FROM @tables;",
        "EXECUTE stmt;",
        "DEALLOCATE PREPARE stmt;"
    ]


class MYSQLHandler(DBHandler):

    def __init__(self, db_config: dict):
        self.db_engine = "mysql"
        self.db_config = db_config

    def drop_all_tables(self):
        return self.execute(drop_all_tables_query)

    def create_schema_statements(self, schema, excluded_columns):
        excluded_columns = excluded_columns or set()
        create_statements = []

        for table in schema.tables:
            table_name = table.table
            columns = [
                f"{column.column} {column.data_type}"
                for column in table.columns
                if column.column not in excluded_columns
            ]

            columns_str = ",\n    ".join(columns)
            create_statement = f"CREATE TABLE {table_name} (\n    {columns_str}\n);"
            create_statements.append(create_statement)

        return create_statements

    def create_insert_statements(self, data_directory):
        table_inserts = {}

        for filename in os.listdir(data_directory):
            if filename.endswith(".csv"):
                table_name = filename[:-4]

                if table_name not in table_inserts:
                    table_inserts[table_name] = []

                with open(os.path.join(data_directory, filename), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        values = ", ".join([f"{value}" for value in row])
                        table_inserts[table_name].append(f"({values})")

        insertion_strings = []
        for table_name, values_list in table_inserts.items():
            values_str = ",\n".join(values_list)
            insert_statement = f"INSERT INTO `{table_name}` VALUES {values_str};"
            insertion_strings.append(insert_statement)

        return insertion_strings

    def execute(self, queries: List[str]):
        result = None
        error = None
        db_instance = get_database(self.db_config)
        for query in queries:
            result, error = db_instance.execute(query)
            if error:
                print(f"Error while executing query. error: {error}")
        return result, error

