"""A gRPC servicer that handles EvalService requests."""

from collections.abc import AsyncIterator

from absl import flags
from absl import logging
from typing import Awaitable, Callable, Optional
import contextvars
import yaml
import grpc
from util.config import load_yaml_config, config_to_df
from util import get_SessionManager
from dataset.dataset import load_json, load_dataset_from_json
from dataset import evalinput
import generators.models as models
import generators.prompts as prompts
import evaluator.evaluator as evaluator
import reporting.report as report
import reporting.bqstore as bqstore
import reporting.analyzer as analyzer
import databases


import eval_request_pb2
import eval_response_pb2
import eval_service_pb2_grpc

_experiment_config = flags.DEFINE_string(
    "self.experiment_config",
    "configs/base_experiment_service.yaml",
    "Path to the eval execution configuration file.",
)

SESSIONMANAGER = get_SessionManager()

rpc_id_var = contextvars.ContextVar("rpc_id", default="default")


class SessionManagerInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, tag: str, rpc_id: Optional[str] = None) -> None:
        self.tag = tag
        self.rpc_id = rpc_id

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        _metadata = dict(handler_call_details.invocation_metadata)
        if rpc_id_var.get() == "default":
            _metadata = dict(handler_call_details.invocation_metadata)
            rpc_id_var.set(self.decorate(_metadata["client-rpc-id"]))
            SESSIONMANAGER.create_session(rpc_id_var.get())
        else:
            rpc_id_var.set(self.decorate(rpc_id_var.get()))
        return await continuation(handler_call_details)

    def decorate(self, rpc_id: str):
        return f"{self.tag}-{rpc_id}"


class EvalServicer(eval_service_pb2_grpc.EvalServiceServicer):
    """A gRPC servicer that handles EvalService requests."""

    def __init__(self) -> None:
        super().__init__()

        logging.info("EvalBench v1.0.0")

    async def Ping(
        self,
        request: eval_request_pb2.PingRequest,
        context: grpc.ServicerContext,
    ) -> eval_response_pb2.EvalResponse:
        return eval_response_pb2.EvalResponse(response=f"ack")

    async def Connect(
        self,
        request,
        context,
    ) -> eval_response_pb2.EvalResponse:
        return eval_response_pb2.EvalResponse(response=f"ack")

    async def EvalConfig(
        self,
        request,
        context,
    ) -> eval_response_pb2.EvalResponse:
        experiment_config = yaml.safe_load(request.yaml_config.decode("utf-8"))
        session = SESSIONMANAGER.get_session(rpc_id_var.get())
        session["config"] = experiment_config

        # Create the DB
        session["db_config"] = load_yaml_config(experiment_config["database_config"])
        session["model_config"] = load_yaml_config(experiment_config["model_config"])
        return eval_response_pb2.EvalResponse(response=f"ack")

    async def ListEvalInputs(
        self,
        request,
        context,
    ) -> eval_request_pb2.EvalInputRequest:
        session = SESSIONMANAGER.get_session(rpc_id_var.get())
        logging.info("Retrieve: %s.", rpc_id_var.get())
        experiment_config = session["config"]
        dataset_config_json = experiment_config["dataset_config"]

        # Load the dataset
        dataset, database = load_dataset_from_json(
            dataset_config_json, experiment_config
        )
        session["db_config"]["database_name"] = database
        dataset, database = load_dataset_from_json(
            dataset_config_json, experiment_config
        )
        session["db_config"]["database_name"] = database
        for eval_input in dataset:
            yield eval_request_pb2.EvalInputRequest(
                id=f"{eval_input.id}",
                query_type=eval_input.query_type,
                database=eval_input.database,
                nl_prompt=eval_input.nl_prompt,
                dialects=eval_input.dialects,
                golden_sql=eval_input.golden_sql,
                eval_query=eval_input.eval_query,
                setup_sql=eval_input.setup_sql,
                cleanup_sql=eval_input.cleanup_sql,
                tags=eval_input.tags,
                other=eval_input.other,
            )

    async def Eval(
        self,
        request_iterator: AsyncIterator[eval_request_pb2.EvalInputRequest],
        context: grpc.ServicerContext,
    ) -> eval_response_pb2.EvalResponse:

        dataset = []
        async for request in request_iterator:
            input = evalinput.EvalInputRequest(
                id=request.id,
                query_type=request.query_type,
                database=request.database,
                nl_prompt=request.nl_prompt,
                dialects=request.dialects,
                golden_sql=request.golden_sql,
                eval_query=request.eval_query,
                setup_sql=request.setup_sql,
                cleanup_sql=request.cleanup_sql,
                tags=request.tags,
                other=request.other,
            )
            dataset.append(input)
        session = SESSIONMANAGER.get_session(rpc_id_var.get())

        session["db"] = databases.get_database(session["db_config"])
        # Load the Query Generator
        session["model_config"]["database_config"] = session["db_config"]
        session["model_generator"] = models.get_generator(session["model_config"])
        # Load the Prompt Generator
        session["prompt_generator"] = prompts.get_generator(
            session["db"], session["config"]
        )
        session["eval"] = evaluator.Evaluator(
            session["config"],
            session["prompt_generator"],
            session["model_generator"],
            session["db"],
        )

        eval = session["eval"]
        job_id, run_time = eval.evaluate(dataset)
        logging.info(f"Run eval job_id:{job_id} run_time:{run_time} for {len(dataset)} eval entries.")

        config_df = config_to_df(
            job_id,
            run_time,
            session["config"],
            session["model_config"],
            session["db_config"],
        )
        report.store(config_df, bqstore.STORETYPE.CONFIGS)

        results = load_json(f"/tmp/eval_output_{job_id}.json")
        results_df = report.quick_summary(results)
        report.store(results_df, bqstore.STORETYPE.EVALS)

        scores = load_json(f"/tmp/score_result_{job_id}.json")
        scores_df, summary_scores_df = analyzer.analyze_result(
            scores, session["config"]
        )
        summary_scores_df["job_id"] = job_id
        summary_scores_df["run_time"] = run_time
        report.store(scores_df, bqstore.STORETYPE.SCORES)
        report.store(summary_scores_df, bqstore.STORETYPE.SUMMARY)
        return eval_response_pb2.EvalResponse(response=f"ack")
