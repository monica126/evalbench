import json
import uuid
import datetime
import queue
import databases
from databases.lock_table import LockTable
from util import printProgressBar
from work import promptgenwork
from work import sqlgenwork
from work import sqlexecwork
from work import scorework
from mp import mprunner
import concurrent.futures
from dataset.evalinput import EvalInputRequest
from dataset.evaloutput import EvalOutput


class Evaluator:
    def __init__(self, experiment_config, prompt_generator, model_generator, db):
        self.eval_ids = None
        self.experiment_config = experiment_config
        self.prompt_generator = prompt_generator
        self.model_generator = model_generator
        self.db = db
        if "eval_ids" in experiment_config.keys() and len(experiment_config["eval_ids"]) > 0:
            self.eval_ids = experiment_config["eval_ids"]

    def evaluate(self, dataset):
        eval_outputs = []
        scoring_results = []

        self.promptrunner = mprunner.MPRunner(10)
        self.genrunner = mprunner.MPRunner(10)
        self.sqlrunner = mprunner.MPRunner(10)
        self.scoringrunner = mprunner.MPRunner(10)
        job_id = f"{uuid.uuid4()}"
        run_time = datetime.datetime.now()

        dialect = self.db.db_config["db"]
        db_config = self.db.db_config

        for query_type in dataset:
            print(f"Processing {query_type} queries")
            prompt_i = 0
            gen_i = 0
            exec_i = 0
            score_i = 0

            dataset_len = len(dataset[query_type])
            dataset_by_type = dataset[query_type]

            self.promptrunner.futures.clear()
            self.genrunner.futures.clear()
            self.sqlrunner.futures.clear()
            self.scoringrunner.futures.clear()

            lock_table = LockTable()

            if query_type == "dql":
                config = db_config.copy()
                config["user_name"] = "tmp_dql"
                db = databases.get_database(config)
            elif query_type == "dml":
                config = db_config.copy()
                config["user_name"] = "tmp_dml"
                db = databases.get_database(config)
            elif query_type == "ddl":
                db_queue = queue.Queue()
                available_database = lock_table.get_available_databases(dialect=dialect, num_required_databases=min(dataset_len, 10))
                for database_name in available_database:
                    config = db_config.copy()
                    config["database_name"] = database_name
                    db = databases.get_database(config)
                    db_queue.put(db)
                
            for eval_input in dataset_by_type:
                eval_output = EvalOutput(eval_input)
                eval_output["job_id"] = job_id
                eval_output["run_time"] = run_time
                if self.eval_ids is not None and eval_input.id not in self.eval_ids:
                    continue
                work = promptgenwork.SQLPromptGenWork(self.prompt_generator, eval_output)
                self.promptrunner.execute_work(work)

            for future in concurrent.futures.as_completed(self.promptrunner.futures):
                eval_output = future.result()
                prompt_i = prompt_i + 1
                printProgressBar(
                    prompt_i, dataset_len, prefix="Prompts:", suffix="Complete", length=50
                )
                work = sqlgenwork.SQLGenWork(self.model_generator, eval_output)
                self.genrunner.execute_work(work)

            for future in concurrent.futures.as_completed(self.genrunner.futures):
                eval_output = future.result()
                gen_i = gen_i + 1
                printProgressBar(
                    gen_i, dataset_len, prefix="SQLGen:", suffix="Complete", length=50
                )
                if query_type == "ddl":
                    db = db_queue.get()

                work = sqlexecwork.SQLExecWork(db, eval_output)
                self.sqlrunner.execute_work(work)

                if query_type == "ddl":
                    db_queue.put(db)

            for future in concurrent.futures.as_completed(self.sqlrunner.futures):
                eval_output = future.result()
                exec_i = exec_i + 1
                work = scorework.ScorerWork(
                    self.experiment_config, eval_output, scoring_results
                )
                self.scoringrunner.execute_work(work)
                printProgressBar(
                    exec_i, dataset_len, prefix="SQLExec:", suffix="Complete", length=50
                )

            for future in concurrent.futures.as_completed(self.scoringrunner.futures):
                eval_output = future.result()
                score_i = score_i + 1
                printProgressBar(
                    score_i, dataset_len, prefix="Scoring:", suffix="Complete", length=50
                )
                eval_outputs.append(eval_output)

            if query_type == "ddl":
                lock_table.release_databases(databases=available_database, dialect=dialect)

        with open(f"/tmp/eval_output_{job_id}.json", "w") as f:
            json.dump(eval_outputs, f, sort_keys=True, indent=4, default=str)

        with open(f"/tmp/score_result_{job_id}.json", "w") as f:
            json.dump(scoring_results, f, sort_keys=True, indent=4, default=str)
        return job_id, run_time
