import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from expo_ft.agents.alg.expo_ft import EXPOLearner
from expo_ft.agents.alg.finite_horizon import canonical_actions,masked_edit_sample,critic_step,edit_temperature_step
from expo_ft.distributions.tanh_transformed import TanhTransformedDistribution
from openpi.training import sharding


def core():
    mesh=sharding.make_mesh(1)
    return EXPOLearner.create(7,jnp.zeros((1,16,16,6)),jnp.zeros(8),jnp.zeros(16),
        actor=None,actor_train_state=None,target_actor_params=None,resume=True,mesh=mesh,
        data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
        replicated_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec()),
        hidden_dims=(16,16),latent_dim_image=8,latent_dim_state=4,encoder_stage_sizes=(1,),encoder_num_filters=4,
        replan_steps=16,action_horizon=16,N=2,n_edit_samples=2,edit_scale=.2,num_qs=3,num_min_qs=2,
        discount=1.,freeze_critic_encoder=False)


def batch():
    mask=jnp.ones((2,128),bool).at[1,64:].set(False)
    return {'observations':jnp.ones((2,16,16,6))*.2,'next_observations':jnp.ones((2,16,16,6))*.3,
        'critic_states':jnp.zeros((2,16)),'next_critic_states':jnp.zeros((2,16)),
        'actions':jnp.ones((2,128))*.1,'action_mask':mask,
        'next_action_mask':jnp.ones((2,128),bool).at[1].set(False),
        'discounts':jnp.array([1.,0.]),'rewards':jnp.array([0.,1.]),'valids':jnp.ones(2)}


def test_mask_gripper_and_tail_gradients():
    a=jnp.ones((2,128))*4
    mask=jnp.ones((2,128),bool).at[1,64:].set(False)
    got=canonical_actions(a,mask,jnp.array([-1.,1.]))
    assert np.all(got[1,64:]==0) and np.all(got[:,7:64:8]==1)
    gradient=jax.grad(lambda x:canonical_actions(x,mask,jnp.array([-1.,1.])).sum())(a)
    assert np.all(gradient[1,64:]==0) and np.all(gradient[:,7:64:8]==0)


def test_masked_log_density_matches_official_distribution_without_mask():
    dist=TanhTransformedDistribution(tfp.distributions.MultivariateNormalDiag(jnp.zeros((2,128)),
                                                                            jnp.full((2,128),.2)))
    edit,logp=masked_edit_sample(dist,jax.random.key(7),jnp.ones((2,128),bool),.2)
    expected=dist.log_prob(edit/.2)-128*jnp.log(.2)
    np.testing.assert_allclose(logp,expected,rtol=1e-6,atol=5e-5)
    empty,none=masked_edit_sample(dist,jax.random.key(7),jnp.zeros((2,128),bool),.2)
    np.testing.assert_array_equal(empty,jnp.zeros((2,128)))
    np.testing.assert_array_equal(none,jnp.zeros(2))


def test_official_network_updates_without_actor_and_ignores_unexecuted_tail():
    learner=core()
    data=batch()
    candidates=jnp.zeros((2,2,128))
    new,info=critic_step(learner,data,candidates,jnp.array([-1.,1.]))
    jax.block_until_ready(info)
    assert int(new.critic.step)==1 and int(new.batch_encoder.step)==1
    assert new.actor_train_state is None and new.target_actor_params is None
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(info))
    changed={**data,'actions':data['actions'].at[1,64:].set(9000.)}
    alternate,other_info=critic_step(learner,changed,candidates.at[1].set(-9000.),jnp.array([-1.,1.]))
    for left,right in zip(jax.tree.leaves(new),jax.tree.leaves(alternate),strict=True):
        np.testing.assert_array_equal(left,right)
    edited,edit_info=edit_temperature_step(new,data,jnp.array([-1.,1.]))
    edited2,_=edit_temperature_step(new,changed,jnp.array([-1.,1.]))
    for left,right in zip(jax.tree.leaves(edited),jax.tree.leaves(edited2),strict=True):
        np.testing.assert_array_equal(left,right)
    assert int(edited.edit_actor.step)==1 and int(edited.temp.step)==1
    assert int(edited.critic.step)==1 and edited.actor_train_state is None
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(edit_info))
