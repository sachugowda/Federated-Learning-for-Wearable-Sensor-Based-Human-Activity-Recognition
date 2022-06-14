import flwr as fl
import sys
import numpy as np
from typing import List, Tuple, Optional, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import EvaluateRes

#class SaveModelStrategy(fl.server.strategy.FedAvg):
    #def aggregate_fit(self,rnd,results,failures):
        #aggregated_weights = super().aggregate_fit(rnd, results, failures)
        #if aggregated_weights is not None:
            # Save aggregated_weights
            #print(f"Saving round {rnd} aggregated_weights...")
           # np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
       # return aggregated_weights
       
       
class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round_weights/round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        #wandb.log({"round": rnd, "server_aggregated_accuracy": accuracy_aggregated})
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}"
        )

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)       

# Create strategy and run server
#strategy = SaveModelStrategy()

strategy = AggregateCustomMetricStrategy(fraction_fit=0.5, fraction_eval=0.5,)

# Start Flower server for three rounds of federated learning with 1Gb of data
fl.server.start_server(
        server_address = "[::]:8080" , 
        config={"num_rounds": 100} ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
        
      
)
