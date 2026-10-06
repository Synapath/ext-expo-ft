from types import SimpleNamespace
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
from openpi.training.utils import TrainState
from expo_ft.agents.vla.partial_ema import train_step_partial_ema,published_params


class TinyModel(nnx.Module):
    def __init__(self):
        self.frozen=nnx.Param(jnp.array([[.09716796875]],dtype=jnp.bfloat16))
        self.trainable=nnx.Param(jnp.array([[.2]],dtype=jnp.float32))

    def compute_loss(self,rng,obs,actions,*,train):
        return (obs@self.trainable.value+self.frozen.value-actions)**2


def test_official_bc_step_preserves_frozen_leaves_and_publishes_full_ema():
    model=TinyModel()
    graph,params=nnx.split(model)
    trainable=lambda path,value: path[0]=='trainable'
    tx=optax.adamw(1e-3)
    state=TrainState(step=0,params=params,model_def=graph,tx=tx,
        opt_state=tx.init(params.filter(trainable)),ema_decay=.99,
        ema_params=params.filter(trainable),ema_trainable_only=True)
    config=SimpleNamespace(trainable_filter=trainable,lr_schedule=SimpleNamespace(create=lambda:lambda step:1e-3))
    update=jax.jit(lambda state:train_step_partial_ema(config,jax.random.key(9),state,
                                                    (jnp.ones((2,1)),jnp.ones((2,1)))))
    new,info=update(state)
    assert int(new.step)==1 and float(info['actor_loss'])>0
    np.testing.assert_array_equal(new.params['frozen'].value,state.params['frozen'].value)
    assert not np.array_equal(new.params['trainable'].value,state.params['trainable'].value)
    published=published_params(new)
    np.testing.assert_array_equal(published['frozen'].value,state.params['frozen'].value)
    np.testing.assert_allclose(published['trainable'].value,
        .99*state.params['trainable'].value+.01*new.params['trainable'].value,rtol=1e-6)
    assert set(new.ema_params)=={'trainable'}
