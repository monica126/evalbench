from .generator import PromptGenerator
from databases import DB

SQLITE_PROMPT_TEMPLATE_WITH_RULES = """You are a SQLite SQL expert.

The database structure is defined by the following table schemas
 (comments after '--' provide additional column descriptions):

**************************
{SCHEMA}
**************************

Please generate a SQLite SQL query for the following question following the rules:
- Output the query only without any explanation.
- Wrap the final generated query in
```sql
QUERY HERE
```
- Always use quotes around the table and column names.

Additional rules to generate correct SQLite SQL dialect:

- Use Table Aliases: Employ aliases to prevent conflicts with duplicate table names.
- Strict Adherence to Schema: You cannot make up any tables or columns that are not explicitly
listed in the provided schema.
- Manage Nulls with ORDER BY: Employ `NULLS LAST` in ORDER BY clauses to control the placement
of null values.
- Arithmetic Operators: Prioritize the use of basic arithmetic operators (+, -, *, /) for
calculations whenever possible.
Only use specialized SQLite functions if absolutely necessary for the desired results.
- Follow SQLite Dialect: Adhere to SQLite syntax, data types, and available functions.
- Choose Join Types Carefully: Select appropriate join types (INNER, LEFT, RIGHT, FULL OUTER)
based on desired relationships.
- Booleans in HAVING: Employ valid boolean expressions within the HAVING clause.
- Type-Aware Functions: Select SQLite functions compatible with the data types in use.
- Cast for Compatibility: Cast data types when necessary for function compatibility or specific manipulations.

Think step by step about generating correct SQLite SQL result!

**************************

Here is the natural language question for generating SQL:
{USER_PROMPT}
"""


PG_PROMPT_TEMPLATE_WITH_RULES = """You are a Postgres SQL expert.

The database structure is defined by the following table schemas:

**************************
{SCHEMA}
**************************

Please generate a Postgres SQL query for the following question following these rules:
- Output the query only without any explanation.
- Do not use markdown code blocks around the outputted query.
- Always use quotes around table and column names.

SQL generation rules:
- Use aliases for tables to avoid ambiguity.
- Only use tables and columns from the provided schema.
- Ensure the generated SQL query is valid and executable.
- Choose appropriate join types for relationships between tables.
- Use functions and operators compatible with the data types of columns.

Think step by step about generating a correct Postgres SQL query!

**************************

Here is the natural language question for generating SQL:
{USER_PROMPT}
"""

MYSQL_PROMPT_TEMPLATE_WITH_RULES = """You are a MySQL SQL expert.

The database structure is defined by the following table schemas:

**************************
{SCHEMA}
**************************

Please generate a MySQL SQL query for the following question following these rules:
- Output the query only without any explanation.
- Do not use markdown code blocks around the outputted query.
- Always use quotes around table and column names.

SQL generation rules:
- Use aliases for tables to avoid ambiguity.
- Only use tables and columns from the provided schema.
- Ensure the generated SQL query is valid and executable.
- Choose appropriate join types for relationships between tables.
- Use functions and operators compatible with the data types of columns.

Think step by step about generating a correct MySQL SQL query!

**************************

Here is the natural language question for generating SQL:
{USER_PROMPT}
"""

SQLSERVER_PROMPT_TEMPLATE_WITH_RULES = """You are a SQLServer SQL expert.

The database structure is defined by the following table schemas:

**************************
{SCHEMA}
**************************

Please generate a SQLServer SQL query for the following question following these rules:
- Output the query only without any explanation.
- Do not use markdown code blocks around the outputted query.
- Always use quotes around table and column names.

SQL generation rules:
- Use aliases for tables to avoid ambiguity.
- Only use tables and columns from the provided schema.
- Ensure the generated SQL query is valid and executable.
- Choose appropriate join types for relationships between tables.
- Use functions and operators compatible with the data types of columns.

Think step by step about generating a correct SQLServer SQL query!

**************************

Here is the natural language question for generating SQL:
{USER_PROMPT}
"""

_PROMPTS_BY_DIALECT = {
    "sqlite": SQLITE_PROMPT_TEMPLATE_WITH_RULES,
    "postgres": PG_PROMPT_TEMPLATE_WITH_RULES,
    "mysql": MYSQL_PROMPT_TEMPLATE_WITH_RULES,
    "sqlserver": SQLSERVER_PROMPT_TEMPLATE_WITH_RULES
}

class SQLGenBasePromptGenerator(PromptGenerator):

    def __init__(self, db: DB, promptgenerator_config):
        super().__init__(db, promptgenerator_config)
        self.db = db
        self.base_prompt = _PROMPTS_BY_DIALECT[db.db_type]

    def setup(self):
        self.schema = self.db.get_ddl_from_db()

    def generate(self, prompt):
        return self.base_prompt.format(
            SCHEMA=self.schema, USER_PROMPT=prompt
        )