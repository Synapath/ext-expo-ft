"""Finite-window adapters around the official EXPO networks and TrainStates.

Uses the same REDQ target, critic MSE, edit objective, temperature objective,
encoder optimizer and Polyak update as expo_ft.py. Adds executed-dimension masks,
per-transition discount, separately prepared critic metadata and proposal entropy
over valid dimensions. No VLA optimizer is created or advanced by this module.
Sampling the current VLA remains the caller's responsibility on every TD batch.
"""
import jax
import jax.numpy as jnp
import optax

from expo_ft.agents.alg.expo_ft import batch_encode
from expo_ft.networks import subsample_image_ensemble


def canonical_actions(actions, valid_mask, gripper_bounds):
    """Score/edit the clipped executed gripper and only valid real dimensions."""
    if actions.shape[-1] != 128 or valid_mask.shape[-1] != 128:
        raise ValueError('Expected 16 x 8 real action dimensions')
    shaped=actions.reshape((*actions.shape[:-1],16,8))
    shaped=shaped.at[...,7].set(jnp.clip(shaped[...,7],gripper_bounds[0],gripper_bounds[1]))
    return jnp.where(valid_mask,shaped.reshape(actions.shape),0.)


def masked_edit_sample(distribution,key,valid_mask,scale):
    """Stable tanh proposal log-density, before controller gripper clipping.

Sampling the preimage avoids evaluating atanh on saturated float32 tanh values.
No entropy or gradient is assigned to an unexecuted tail dimension.
"""
    normal=distribution.distribution
    preimage=normal.sample(seed=key)
    sample=distribution.bijector.forward(preimage)
    std=normal.stddev()
    per_dim=-.5*((preimage-normal.mean())/std)**2-jnp.log(std)-.5*jnp.log(2*jnp.pi)
    per_dim-=2*(jnp.log(2.)-preimage-jax.nn.softplus(-2*preimage))
    log_prob=jnp.sum(jnp.where(valid_mask,per_dim-jnp.log(scale),0.),axis=-1)
    return jnp.where(valid_mask,sample*scale,0.),log_prob


def q_values(core,encoded,states,actions,mask,gripper_bounds,*,params=None,sample_num=None):
    actions=canonical_actions(actions,mask,gripper_bounds)
    return core.critic.apply_fn({'params':core.critic.params if params is None else params},
                               encoded,actions,False,p=states,sample_num=sample_num)


def _target_action(core,encoded,states,candidates,mask,gripper_bounds,edit_key,selection_key):
    batch,n,dimensions=candidates.shape
    if n!=core.N or not 0<=core.n_edit_samples<=n:
        raise ValueError('Candidate count/edit count mismatch')
    candidates=canonical_actions(candidates,mask[:,None,:],gripper_bounds)
    count=core.n_edit_samples
    if count:
        base=candidates[:,:count].reshape(batch*count,dimensions)
        repeated_mask=jnp.repeat(mask,count,axis=0)
        dist=core.edit_actor.apply_fn({'params':core.edit_actor.params},jnp.repeat(encoded,count,axis=0),
                                      actions=base,p=jnp.repeat(states,count,axis=0))
        edits,_=masked_edit_sample(dist,edit_key,repeated_mask,core.edit_scale)
        edited=canonical_actions(base+edits,repeated_mask,gripper_bounds).reshape(batch,count,dimensions)
        candidates=jnp.concatenate([candidates,edited],axis=1)
    total=candidates.shape[1]
    params=subsample_image_ensemble(selection_key,core.target_critic.params,core.num_min_qs,core.num_qs)
    qs=q_values(core,jnp.repeat(encoded,total,axis=0),jnp.repeat(states,total,axis=0),
                candidates.reshape(batch*total,dimensions),jnp.repeat(mask,total,axis=0),gripper_bounds,
                params=params,sample_num=core.num_min_qs).min(axis=0).reshape(batch,total)
    index=jnp.argmax(qs,axis=1)
    return candidates[jnp.arange(batch),index],{'selected_edit_fraction':jnp.mean(index>=n),
                                              'selection_q_mean':qs.mean()}


@jax.jit
def critic_step(core,batch,next_candidates,gripper_bounds):
    """One official-style critic/encoder/target update; no actor/BC update."""
    edit_key,selection_key,backup_key,dropout_key,rng=jax.random.split(core.rng,5)
    encoded_next=batch_encode(core.batch_encoder.apply_fn,core.batch_encoder.params,
                              batch['next_observations'],stop_gradient=True)
    next_action,sampling=_target_action(core,encoded_next,batch['next_critic_states'],next_candidates,
        batch['next_action_mask'],gripper_bounds,edit_key,selection_key)
    target_params=subsample_image_ensemble(backup_key,core.target_critic.params,core.num_min_qs,core.num_qs)
    next_qs=q_values(core,encoded_next,batch['next_critic_states'],next_action,batch['next_action_mask'],
                    gripper_bounds,params=target_params,sample_num=core.num_min_qs)
    next_q=next_qs.min(axis=0)
    # No edit entropy appears in the critic target. Do not replace NaNs with 0.
    target=jax.lax.stop_gradient(batch['rewards']+batch['discounts']*next_q)
    params={'critic':core.critic.params}
    if not core.freeze_critic_encoder:
        params['encoder']=core.batch_encoder.params

    def loss_fn(params):
        encoded=batch_encode(core.batch_encoder.apply_fn,params.get('encoder',core.batch_encoder.params),
                             batch['observations'],stop_gradient=core.freeze_critic_encoder)
        actions=canonical_actions(batch['actions'],batch['action_mask'],gripper_bounds)
        qs=core.critic.apply_fn({'params':params['critic']},encoded,actions,True,
                               p=batch['critic_states'],rngs={'dropout':dropout_key})
        loss=jnp.mean((qs-target)**2*batch['valids'])
        return loss,{'critic_loss':loss,'q_mean':qs.mean(),'q_min':qs.min(),'q_max':qs.max(),
                     'target_mean':target.mean(),'target_min':target.min(),'target_max':target.max(),
                     'next_q_nonfinite':jnp.mean(~jnp.isfinite(next_qs))}
    grads,info=jax.grad(loss_fn,has_aux=True)(params)
    critic=core.critic.apply_gradients(grads=grads['critic'])
    encoder=core.batch_encoder if core.freeze_critic_encoder else core.batch_encoder.apply_gradients(grads=grads['encoder'])
    target_core=core.target_critic.replace(params=optax.incremental_update(critic.params,core.target_critic.params,core.tau))
    return core.replace(critic=critic,batch_encoder=encoder,target_critic=target_core,rng=rng),{
        **info,**sampling,'critic_grad_norm':optax.global_norm(grads['critic'])}


@jax.jit
def edit_temperature_step(core,batch,gripper_bounds):
    """One edit and temperature update, masking both Q input and proposal entropy."""
    sample_key,dropout_key,rng=jax.random.split(core.rng,3)
    encoded=batch_encode(core.batch_encoder.apply_fn,core.batch_encoder.params,batch['observations'],stop_gradient=True)
    base=canonical_actions(batch['actions'],batch['action_mask'],gripper_bounds)
    def loss_fn(params):
        dist=core.edit_actor.apply_fn({'params':params},encoded,actions=base,p=batch['critic_states'])
        edits,logp=masked_edit_sample(dist,sample_key,batch['action_mask'],core.edit_scale)
        actions=canonical_actions(base+edits,batch['action_mask'],gripper_bounds)
        qs=core.critic.apply_fn({'params':core.critic.params},encoded,actions,True,
                               p=batch['critic_states'],rngs={'dropout':dropout_key})
        loss=(core.entropy_scale*logp*core.temp.apply_fn({'params':core.temp.params})-qs.mean(axis=0)).mean()
        return loss,{'edit_loss':loss,'edit_q':qs.mean(),'entropy_per_row':-logp,
                     'entropy':-logp.mean(),'edit_l2':jnp.linalg.norm(edits,axis=-1).mean()}
    grads,info=jax.grad(loss_fn,has_aux=True)(core.edit_actor.params)
    actor=core.edit_actor.apply_gradients(grads=grads)
    entropy=jax.lax.stop_gradient(info['entropy_per_row'])
    target=core.target_entropy*jnp.sum(batch['action_mask'],axis=-1)/128
    def temperature_loss(params):
        temperature=core.temp.apply_fn({'params':params})
        return temperature*(entropy-target).mean()
    temp_grads=jax.grad(temperature_loss)(core.temp.params)
    temp=core.temp.apply_gradients(grads=temp_grads)
    return core.replace(edit_actor=actor,temp=temp,rng=rng),{
        **info,'temperature':temp.apply_fn({'params':temp.params}),'temperature_loss':temperature_loss(core.temp.params)}


@jax.jit
def empirical_return_step(core,batch,returns,gripper_bounds):
    """Diagnostic policy evaluation regression, NOT an EXPO TD/RL update.

The caller supplies the observed return of the recorded SFT continuation.
This permits a like-for-like comparison with SFT-continuation probe outcomes.
It does not train an edit policy or claim to evaluate an improved OTF policy.
"""
    key,rng=jax.random.split(core.rng)
    params={'critic':core.critic.params,'encoder':core.batch_encoder.params}
    def loss_fn(params):
        encoded=batch_encode(core.batch_encoder.apply_fn,params['encoder'],batch['observations'],stop_gradient=False)
        qs=core.critic.apply_fn({'params':params['critic']},encoded,
            canonical_actions(batch['actions'],batch['action_mask'],gripper_bounds),True,
            p=batch['critic_states'],rngs={'dropout':key})
        loss=jnp.mean((qs-returns)**2)
        return loss,{'empirical_return_mse':loss,'q_mean':qs.mean(),'q_min':qs.min(),'q_max':qs.max(),
                     'return_mean':returns.mean()}
    grads,info=jax.grad(loss_fn,has_aux=True)(params)
    critic=core.critic.apply_gradients(grads=grads['critic'])
    encoder=core.batch_encoder.apply_gradients(grads=grads['encoder'])
    target=core.target_critic.replace(params=optax.incremental_update(critic.params,core.target_critic.params,core.tau))
    return core.replace(critic=critic,batch_encoder=encoder,target_critic=target,rng=rng),info
