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
        print(_metadata["client-rpc-id"])
        print(rpc_id_var.get())
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

        # self.experiment_config = load_yaml_config(_experiment_config.value)
        # if self.experiment_config == "":
        #     return

        # logging.info("Loaded %s", _experiment_config.value)

        # db_config_yaml = self.experiment_config["database_config"]
        # model_config_yaml = self.experiment_config["model_config"]
        # dataset_config_json = self.experiment_config["dataset_config"]

        # # Load the dataset
        # dataset, database = load_dataset_from_json(dataset_config_json)

        # # Load the model config
        # self.model_config = load_yaml_config(model_config_yaml)

        # # Load the DB config DB name comes from the dataset
        # self.db_config = load_yaml_config(db_config_yaml)
        # self.db_config["database_name"] = database
        # self.model_config["database_config"] = self.db_config

        # # Create the DB
        # self.db = databases.get_database(self.db_config)

        # # Load the Query Generator
        # self.model_generator = models.get_generator(self.model_config)

        # # Load the Prompt Generator
        # self.prompt_generator = prompts.get_generator(self.db, self.experiment_config)

    # async def Eval(
    #     self,
    #     request_iterator: AsyncIterator[eval_request_pb2.EvalRequest],
    #     context: grpc.ServicerContext,
    # ) -> eval_response_pb2.EvalResponse:
    #     eval = evaluator.Evaluator(
    #         self.experiment_config, self.prompt_generator, self.model_generator, self.db
    #     )
    #     dataset = []
    #     async for request in request_iterator:
    #         input = evalinput.EvalInput(
    #             id=request.id,
    #             query_type=request.query_type,
    #             database=request.database,
    #             # TODO: This is a hack to allow client side SQLGen generated SQL
    #             # to pass through PromptGen and SQLGen generators, revisit this.
    #             nl_prompt=request.eval_query,
    #             dialects=request.dialects,
    #             golden_sql=request.golden_sql,
    #             eval_query=request.eval_query,
    #             setup_sql=request.setup_sql,
    #             cleanup_sql=request.cleanup_sql,
    #             tags=request.tags,
    #         )
    #         dataset.append(input)

    #     job_id, run_time = eval.evaluate(dataset)
    #     results = load_json("/tmp/eval_output.json")
    #     results_df = report.quick_summary(results)

    #     scores = load_json("/tmp/score_result.json")
    #     scores_df, summary_scores_df = analyzer.analyze_result(
    #         scores, self.experiment_config
    #     )
    #     summary_scores_df["job_id"] = job_id
    #     summary_scores_df["run_time"] = run_time
    #     return eval_response_pb2.EvalResponse(response=f"ack")

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
        config = yaml.safe_load(request.yaml_config.decode("utf-8"))
        session = SESSIONMANAGER.get_session(rpc_id_var.get())
        session["config"] = config
        logging.info("Config: %s. Keys: %s", rpc_id_var.get(), session.keys())
        session = SESSIONMANAGER.get_session(rpc_id_var.get())
        logging.info("Config: %s. Keys: %s", rpc_id_var.get(), session.keys())
        return eval_response_pb2.EvalResponse(response=f"ack")

    async def ListEvalInputs(
        self,
        request,
        context,
    ) -> eval_response_pb2.EvalInput:
        session = SESSIONMANAGER.get_session(rpc_id_var.get())
        logging.info("Retrieve: %s. Keys: %s", rpc_id_var.get(), session.keys())
        experiment_config = session["config"]
        dataset_config_json = experiment_config["dataset_config"]

        # Load the dataset
        dataset, database = load_dataset_from_json(dataset_config_json)
        for eval_input in dataset:
            yield eval_response_pb2.EvalInput(
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
            )
