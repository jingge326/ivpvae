from model.ivp_solvers import CouplingFlow, ODEModel, ResNetFlow, GRUFlow
from model.components import Embedding_MLP, Embedding_MLP, Reconst_Mapper_MLP
from model.ivp_vae_biclass import IVPVAE_BiClass
from model.ivp_vae_extrap import IVPVAE_Extrap
from utils import SolverWrapper


class ModelFactory:
    def __init__(self, args):
        self.args = args

    def build_ivp_solver(self, states_dim):
        ivp_solver = None
        hidden_dims = [self.args.hidden_dim] * self.args.hidden_layers
        if self.args.ivp_solver == 'ode':
            ivp_solver = SolverWrapper(ODEModel(states_dim, self.args.odenet, hidden_dims, self.args.activation,
                                                self.args.final_activation, self.args.ode_solver, self.args.solver_step, self.args.atol, self.args.rtol))
        else:
            if self.args.ivp_solver == 'couplingflow':
                flow = CouplingFlow
            elif self.args.ivp_solver == 'resnetflow':
                flow = ResNetFlow
            elif self.args.ivp_solver == 'gruflow':
                flow = GRUFlow
            else:
                raise NotImplementedError

            ivp_solver = SolverWrapper(flow(
                states_dim, self.args.flow_layers, hidden_dims, self.args.time_net, self.args.time_hidden_dim))
        return ivp_solver

    def init_components(self):

        embedding_nn = Embedding_MLP(
            self.args.variable_num, self.args.latent_dim)

        ivp_solver = self.build_ivp_solver(self.args.latent_dim)

        reconst_mapper = Reconst_Mapper_MLP(
            self.args.latent_dim, self.args.variable_num)

        return embedding_nn, ivp_solver, reconst_mapper

    def initialize_biclass_model(self):

        embedding_nn, diffeq_solver, reconst_mapper = self.init_components()

        return IVPVAE_BiClass(
            args=self.args,
            embedding_nn=embedding_nn,
            reconst_mapper=reconst_mapper,
            diffeq_solver=diffeq_solver)

    def initialize_extrap_model(self):

        embedding_nn, diffeq_solver, reconst_mapper = self.init_components()

        return IVPVAE_Extrap(
            args=self.args,
            embedding_nn=embedding_nn,
            reconst_mapper=reconst_mapper,
            diffeq_solver=diffeq_solver)
