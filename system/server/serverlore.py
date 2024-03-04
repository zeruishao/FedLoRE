from system.server.serverbase import Server
from system.client.clientlore import clientLoRE


class FedLoRE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientLoRE)
        self.pre_round = args.pre_round
        self.cur_round = 0
        print("Finished creating server and clients.")


    def train(self):
        for i in range(self.global_rounds + 1):
            self.cur_round = i
            self.send_models()

            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate personalized models")
            self.evaluate()

            for client in self.clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            if self.cur_round < self.pre_round:
                client.set_parameters(self.global_model)
            else:
                client.set_parameters_lora(self.global_model)


    def receive_models(self):
        self.uploaded_ids = []
        for client in self.clients:
            self.uploaded_ids.append(client.id)


    def aggregate_parameters(self):
        key_sum, key_count = {}, {}
        for cid in self.uploaded_ids:
            if self.cur_round < self.pre_round:
                for key, value in self.clients[cid].model.state_dict().items():
                    if key not in key_sum:
                        key_sum[key], key_count[key] = 0, 0
                    key_sum[key] += value
                    key_count[key] += 1
            else:
                for key, value in self.clients[cid].lowrank_gradient_dict.items():
                    if key not in key_sum:
                        key_sum[key], key_count[key] = 0, 0
                    key_sum[key] += value
                    key_count[key] += 1
        for key in key_sum.keys():
            self.global_model[key] = key_sum[key] / key_count[key]