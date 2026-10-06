"""EXPO BC with a trainable-only EMA, preserving frozen encoder leaves exactly."""
import dataclasses
import jax
from openpi.training.checkpoints import inference_params
from expo_ft.agents.vla.pi05 import train_step


def train_step_partial_ema(config,rng,state,batch):
    if not state.ema_trainable_only or state.ema_decay is None or state.ema_params is None:
        raise ValueError('This path requires an explicit trainable-only EMA')
    # Reuse the official flow-matching, trainable filter, optimizer and RNG path.
    # Do not run EMA arithmetic on BF16 frozen leaves: even x*.99+x*.01 may round.
    raw_state=dataclasses.replace(state,ema_decay=None,ema_params=None,ema_trainable_only=False)
    updated,info=train_step(config,rng,raw_state,batch)
    trainable=updated.params.filter(config.trainable_filter)
    ema=jax.tree.map(lambda old,new:state.ema_decay*old+(1-state.ema_decay)*new,state.ema_params,trainable)
    updated=dataclasses.replace(updated,ema_decay=state.ema_decay,ema_params=ema,ema_trainable_only=True)
    return updated,info


def published_params(state):
    """Compose current full EMA, never substituting target VLA parameters."""
    return inference_params(state)
