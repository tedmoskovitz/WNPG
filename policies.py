import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
import scipy.stats

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs, assumed to be gray scale [0,1]
        with tf.variable_scope("model", reuse=reuse):
            if nh <= 20:
                h = conv(X, 'c1', nf=8, rf=4, stride=2, init_scale=np.sqrt(2))
                h = conv(h, 'c2', nf=16, rf=2, stride=2, init_scale=np.sqrt(2))
                h = conv(h, 'c3', nf=16, rf=1, stride=1, init_scale=np.sqrt(2))
            elif nh <= 40:
                h = conv(X, 'c1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2))
                h = conv(h, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
                h = conv(h, 'c3', nf=64, rf=2, stride=1, init_scale=np.sqrt(2))
            else:
                h = conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                h = conv(h, 'c2', nf=64, rf=4, stride=3, init_scale=np.sqrt(2))
                h = conv(h, 'c3', nf=64, rf=2, stride=2, init_scale=np.sqrt(2))                
            h = conv_to_fc(h)

            # fully connected layer
            h1 = fc(h, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            #h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h1, 'pi', actdim, act=lambda x:x, init_scale=0.01)
            h1 = fc(h, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            #h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h1, 'vf', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, M=None): #pylint: disable=W0613
        assert M is not None
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        act = tf.tanh
        with tf.variable_scope("model", reuse=reuse):
            h1 = act(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = act(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = act(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = act(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        # reparameterize actions
        noise = tf.random_normal([nbatch, M, actdim])
        mu = tf.expand_dims(pi, axis=1)
        std = tf.expand_dims(tf.exp(pi * 0.0 + logstd), axis=1)
        a_reparameterized = mu + std * noise

        # sample actions
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, batchactions = sess.run([a0, vf, neglogp0, a_reparameterized], {X:ob})
            return a, v, self.initial_state, neglogp, batchactions

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})


        self.a0 = a0
        self.X = X
        self.pi = pi 
        self.vf = vf
        self.step = step
        self.value = value
        self.a_reparameterized = a_reparameterized


class DistMlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, K=32, reuse=False, M=None): #pylint: disable=W0613
        assert M is not None
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        act = tf.tanh
        with tf.variable_scope("model", reuse=reuse):
            h1 = act(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = act(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = act(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = act(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', K) #[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())


        # reparameterize actions
        noise = tf.random_normal([nbatch, M, actdim])
        mu = tf.expand_dims(pi, axis=1)
        std = tf.expand_dims(tf.exp(pi * 0.0 + logstd), axis=1)
        a_reparameterized = mu + std * noise

        # sample actions
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        # distributional info
        self.K = K
        vf_mean = tf.reduce_mean(vf, axis=-1)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, batchactions, v_avg = sess.run([a0, vf, neglogp0, a_reparameterized, vf_mean], {X:ob})
            return a, v, self.initial_state, neglogp, batchactions, v_avg

        def value(ob, *_args, **_kwargs):
            return sess.run(vf_mean, {X:ob})

        self.a0 = a0
        self.X = X
        self.pi = pi 
        self.vf = vf
        self.vf_mean = vf_mean
        self.step = step
        self.value = value
        self.a_reparameterized = a_reparameterized



class TinyMlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, M=None): #pylint: disable=W0613
        assert M is not None
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        act = tf.tanh
        with tf.variable_scope("model", reuse=reuse):
            h1 = act(fc(X, 'pi_fc1', nh=12, init_scale=np.sqrt(2)))
            h2 = act(fc(h1, 'pi_fc2', nh=12, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = act(fc(X, 'vf_fc1', nh=12, init_scale=np.sqrt(2)))
            h2 = act(fc(h1, 'vf_fc2', nh=12, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        # reparameterize actions
        noise = tf.random_normal([nbatch, M, actdim])
        mu = tf.expand_dims(pi, axis=1)
        std = tf.expand_dims(tf.exp(pi * 0.0 + logstd), axis=1)
        a_reparameterized = mu + std * noise

        # sample actions
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, batchactions = sess.run([a0, vf, neglogp0, a_reparameterized], {X:ob})
            return a, v, self.initial_state, neglogp, batchactions

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})


        self.a0 = a0
        self.X = X
        self.pi = pi 
        self.vf = vf
        self.step = step
        self.value = value
        self.a_reparameterized = a_reparameterized
