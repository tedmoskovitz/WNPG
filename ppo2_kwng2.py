import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.a2c.utils import discount_with_dones
from kwng_tf import KWNG
from kwng_tf_mean import KWNG as KWNG_mean
from kwng_tf_long import KWNG as KWNG_long
from gaussian_tf import Gaussian
import pdb

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, D, omega, bias, M, gamma_W, wasserstein_coeff, KWNG_estimator, base_opt):

        # D: number of random features
        # omega: [D, actdim] random feature vector
        # bias: [D, 1] random featture vector
        # M: number of actions per state
        # gamma_W: gamma for wasserstein distance

        """
        need placeholder for old loss
        """


        act_dim = ac_space.high.size

        sess = tf.get_default_session()
        self.sess = sess
        self.KWNG_estimator = KWNG_estimator
        self.clip_grad = max_grad_norm is not None
        self.clip_value = max_grad_norm
        self.old_loss = tf.Variable(-1., name='old_loss', trainable=False, dtype=tf.float32)
        self.dot_prod = tf.Variable(0., name='dot_prod', trainable=False, dtype=tf.float32)
        self.reduction_coeff = tf.Variable(0.85, name='reduction_coeff', trainable=False, dtype=tf.float32)#reduction_coeff
        self.dumping_freq = 5 ##dumping_freq
        self.min_red = 0.25 #min_red
        self.max_red = 0.75 #max_red
        self.eps_min = 1e-10
        self.eps_max = 1e5
        self.dumping_counter = tf.Variable(0, name='dumping_counter', trainable=False, dtype=tf.int32) #0
        self.reduction_factor = tf.Variable(0., name='reduction_factor', trainable=False, dtype=tf.float32)


        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # setup Wasserstein trust region here
        A_reparameterized = train_model.a_reparameterized
        BETA_1 = tf.placeholder(tf.float32, [None, D])
        BETA_2 = tf.placeholder(tf.float32, [None, D])

        def get_random_features(x):
            nsample = x.shape[1]
            return np.cos(np.dot(x, omega.T) + bias.T) * np.sqrt(2. / D)

        def get_random_features_tf(x):
            omega_tf = tf.constant(omega.T, dtype=tf.float32)
            bias_tf = tf.constant(bias.T, dtype=tf.float32)
            return tf.cos(tf.matmul(x, omega_tf) + bias_tf) * np.sqrt(2. / D)

        # random feature for parameterized action
        A_reparameterized_random_features = tf.reshape(get_random_features_tf(tf.reshape(A_reparameterized, [-1, act_dim])), [-1, M, D])

        # placeholder for reference actions
        target_actions = tf.placeholder(tf.float32, [None, M, act_dim])
        target_actions_random_features = get_random_features_tf(tf.reshape(target_actions, [-1, act_dim]))
        target_actions_random_features = tf.reshape(target_actions_random_features, [-1, M, D])

        distance = tf.reduce_sum(tf.square(target_actions - A_reparameterized), axis=-1)

        test_1 = tf.reduce_sum(tf.expand_dims(BETA_1, axis=1) * A_reparameterized_random_features, axis=-1)
        test_2 = tf.reduce_sum(tf.expand_dims(BETA_2, axis=1) * target_actions_random_features, axis=-1)

        # test_1, test_2, distance should have the same dimension [nbatch, M]
        weight = test_1 + test_2 - distance
        wasserstein_distances = test_1 + test_2 - gamma_W * tf.exp(weight / gamma_W)
        wasserstein_distance = tf.reduce_mean(wasserstein_distances)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + wasserstein_distance * wasserstein_coeff
        ## adjust epsilon ##  change reduction factor and epsilon, compute conditioning matrix 
        dumping_ops = self.dumping(loss, LR) 
        act_embed = tf.reshape(A_reparameterized, [-1, M*act_dim]) 
        v_embed = tf.expand_dims(vpred, -1)
        output = tf.concat([act_embed, v_embed], axis=-1)
        print ("EMBEDDING SHAPE: ", output.get_shape().as_list())
        self.KWNG_estimator.compute_cond_matrix(train_model, output) ###PICKED behav policy embedding### 
        ####################

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)

        ## compute natural gradient ##
        g = [tf.reshape(gr, [-1]) for gr in grads]
        g = tf.concat(g, axis=0) # flattened gradients
        cond_g = self.KWNG_estimator.compute_natural_gradient(g)

        # if the dot product is negative, just use the euclidean grad
        self.dot_prod = tf.assign(self.dot_prod, tf.reduce_sum(g * cond_g))
        cond_g = tf.cond(self.dot_prod <= 0, lambda: g, lambda: cond_g) 
        # grad clipping by norm
        if self.clip_grad:
            cond_g, self.dot_prod = self.clip_gradient(cond_g)
        # save old loss
        self.old_loss = tf.assign(self.old_loss, loss) 
        # reshape gradients
        start = 0; cond_g_list = []; 
        for gr in grads:
            gr_shape = gr.get_shape().as_list()
            gr_size = np.prod(gr_shape)
            cond_g_list.append(tf.reshape(cond_g[start : start + gr_size], gr_shape)) 
            start += gr_size
        grads = cond_g_list
        ##############################

        grads = list(zip(grads, params))

        opt_dict = {
            'adam': tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5),
            'momentum': tf.train.MomentumOptimizer(learning_rate=LR, momentum=0.9),
            'sgd': tf.train.GradientDescentOptimizer(learning_rate=LR)
        }
        trainer = opt_dict[base_opt] 
        _train = trainer.apply_gradients(grads)

        self.beta_1 = np.zeros([nbatch_train, D])
        self.beta_2 = np.zeros([nbatch_train, D])

        def clear_beta():
            self.beta_1 = np.zeros([nbatch_train, D])
            self.beta_2 = np.zeros([nbatch_train, D]) 
               

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, oldbatchactions, states=None, T=1, lr_beta=0.0):
            # oldbatchactions: old actions executed by old policy, M action per state
            # beta_1: param for test function 1
            # beta_2: param for test function 2
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # update test function
            # get random features for oldbatchactions
            oldbatchactions_randomfeatures = np.reshape(get_random_features(np.reshape(oldbatchactions, [-1, act_dim])), [-1, M, D])
            for t in range(T):
                # sample new actions
                newbatchactions = sess.run(A_reparameterized, feed_dict={train_model.X:obs}) # [nbatch, M, act_dim]
                # get random features
                newbatchactions_randomfeatures = np.reshape(get_random_features(np.reshape(newbatchactions, [-1, act_dim])), [-1, M, D])
                # distance
                dist = np.sum((newbatchactions - oldbatchactions)**2, axis=-1) # [nbatch, M]
                # test functions
                weight1 = np.sum(np.expand_dims(self.beta_1, axis=1) * newbatchactions_randomfeatures, axis=-1) # [nbatch, M]
                weight2 = np.sum(np.expand_dims(self.beta_2, axis=1) * oldbatchactions_randomfeatures, axis=-1) # [nbatch, M]
                coeff = 1.0 - np.exp((weight1 + weight2 - dist) / gamma_W) # [nbatch, M]
                # compute gradient
                update1 = np.mean(np.expand_dims(coeff, axis=2) * newbatchactions_randomfeatures, axis=1) # [nbatch, D]
                update2 = np.mean(np.expand_dims(coeff, axis=2) * oldbatchactions_randomfeatures, axis=1) # [nbatch, D]
                # update beta
                self.beta_1 += lr_beta * update1
                self.beta_2 += lr_beta * update2

            # update policy
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, 
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, BETA_1:self.beta_1, BETA_2:self.beta_2, target_actions:oldbatchactions}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac] + list(dumping_ops) + [self.dot_prod, self.old_loss, _train],  #dumping_counter_op, 
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.clear_beta = clear_beta
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.sess = sess
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

    def dumping(self, loss, lr):

        # compute reduction ratio
        red = 2. * (self.old_loss - loss) / (lr * self.dot_prod)
        new_reduction_factor = tf.cond(
            tf.logical_and(self.old_loss > -1, red > self.reduction_factor), 
            lambda: red,
            lambda: self.reduction_factor)

        #self.reduction_factor = new_reduction_factor
        reduction_factor_op = tf.assign(self.reduction_factor, new_reduction_factor)

        # increment dumping counter
        self.dumping_counter = tf.assign(self.dumping_counter, self.dumping_counter + 1) 

        base_pred = tf.logical_and(self.old_loss > -1, tf.equal(tf.mod(self.dumping_counter, self.dumping_freq), 0))
        pred1 = tf.logical_and(base_pred, tf.logical_and(self.reduction_factor < self.min_red, self.KWNG_estimator.eps < self.eps_max))
        pred2 = tf.logical_and(base_pred, tf.logical_and(self.reduction_factor > self.max_red, self.KWNG_estimator.eps > self.eps_min))

        # if both conditions in pred1 are true, rescale eps like this
        new_eps = tf.cond(
            tf.cast(pred1, tf.bool),
            lambda: self.KWNG_estimator.eps / self.reduction_coeff,
            lambda: self.KWNG_estimator.eps)

        # if both conditions in pred2 are true, rescale eps like this instead
        new_eps = tf.cond(
            tf.cast(pred2, tf.bool),
            lambda: self.KWNG_estimator.eps * self.reduction_coeff,
            lambda: self.KWNG_estimator.eps)

        # if the base predicate is true at all, reset the reduction factor to 0
        new_reduction_factor = tf.cond(
            tf.cast(base_pred, tf.bool),
            lambda: 0.,
            lambda: new_reduction_factor)

        self.reduction_factor = tf.assign(self.reduction_factor, new_reduction_factor) 
        self.KWNG_estimator.eps = tf.assign(self.KWNG_estimator.eps, new_eps) 

        return self.reduction_factor, self.dumping_counter, self.KWNG_estimator.eps 

    def clip_gradient(self, cond_g):
        
        norm_grad = tf.norm(cond_g)
        clip_coef = self.clip_value / (norm_grad + 1e-6)

        result, new_dot_prod = tf.cond(clip_coef < 1.0,
            lambda: (cond_g / norm_grad, self.dot_prod / norm_grad),
            lambda: (cond_g, self.dot_prod))

        return result, tf.assign(self.dot_prod, new_dot_prod)

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, gae):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.gae = gae

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_batchactions = []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs, batchactions = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)            
            mb_batchactions.append(batchactions)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')

                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_batchactions = np.asarray(mb_batchactions, dtype=np.float32)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        if self.gae is True:
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0        
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    nextvalues = mb_values[t+1]
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
        else:
            mb_returns = discount_with_dones(mb_rewards, mb_dones, self.gamma)
            mb_returns = np.array(mb_returns)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_batchactions)), 
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, callback=None, D=None, M=None, gamma_W=None, wasserstein_coeff=None,
            T=None, lr_beta=None, sigma=None, gae=None, kwng_args=None, base_opt='adam', num_kwng_basis=5, kwng_method='kwng',
            lr_decay=-1, decay_factor=4.):

    assert gae is not None

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # generate random features with rbf kernel
    omega = np.random.randn(D, ac_space.high.size) * 1.0 / sigma
    bias = np.random.rand(D, 1) * 2 * np.pi

    # make model function
    ### define kwng estimator ###
    kernel  = Gaussian(1, 0., dtype=tf.float32, device='gpu') #kwng_args['log_bandwidth']

    kwng_dict = {
        'kwng': KWNG,
        'kwng_mean': KWNG_mean,
        'kwng_long': KWNG_long
    }

    estimator = kwng_dict[kwng_method](kernel,
                     eps=1e-5, 
                     num_basis=num_kwng_basis, 
                     with_diag_mat=1) 
    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, KWNG_estimator=estimator, 
                    max_grad_norm=max_grad_norm, D=D, omega=omega, bias=bias, M=M, gamma_W=gamma_W, wasserstein_coeff=wasserstein_coeff, base_opt=base_opt)

    # make model
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, gae=gae)

    # save model using saver
    saver = tf.train.Saver()
    if callback.arg['continue-train'] == 1:
        saver.restore(model.sess, callback.directory + '/model/model')
    elif callback.arg['continue-train'] == 0:
        pass
    else:
        raise NotImplementedError

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        if lr_decay != -1 and update % lr_decay == 0: lrnow /= decay_factor; 
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, oldbatchactions, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []

        # clear betas
        model.clear_beta()

        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, oldbatchactions))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, T=T, lr_beta=lr_beta))
        else: # recurrent version
            raise NotImplementedError
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))            

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            #logger.logkv('xpos', safemean([epinfo['pos'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
                if lossname == 'policy_entropy':
                    policy_entropy = lossval
                if lossname == 'approxkl':
                    approxkl = lossval
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
        if callback is not None:
            eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
            eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

            total_timesteps_sofar = update * nbatch
            callback(locals(), globals())

    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def render_evaluate(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, callback=None):

    logger.info('rendering')

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    # make model
    model = make_model()

    # save model using saver
    saver = tf.train.Saver()
    logger.info('Restore trained model')
    saver.restore(model.sess, callback.model_dir)
    print('loading model success')

    # load obs_rms
    import pickle
    with open(callback.directory + '/ob_rms.pkl', 'rb') as output:
        ob_rms = pickle.load(output)
    print('loading mean', ob_rms.mean)
    print('loading var', ob_rms.var)

    def filter(ob):
        ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10, 10)
        return ob

    obsdict = []
    assert len(env.venv.envs) == 1
    for e in range(100):
        rsum = 0
        obsrecord = []
        done = False
        obs = env.reset()
        obs = filter(obs)
        env.venv.envs[0].render()
        while not done:
            actions, values, _, neglogpacs = model.step(obs)
            newobs, r, done, _ = env.step(actions)
            obsrecord.append(obs)
            obs = newobs
            obs = filter(obs)
            env.venv.envs[0].render()
            rsum += r
        print('total rewards', rsum)

    if callback is not None:
        pass
        callback(locals(), globals())


def collect_samples_evaluate(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, callback=None):

    logger.info('rendering')

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    ### define kwng estimator ###
    kernel  = Gaussian(1, args['log_bandwidth'], dtype=tr.float32, device=device)
    estimator = KWNG(kernel,
                     eps=args['epsilon'],
                     num_basis=args['num_basis'],
                     with_diag_mat=args['with_diag_mat'])

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, KWNG_estimator=estimator)

    # make model
    model = make_model()

    # save model using saver
    saver = tf.train.Saver()
    logger.info('Restore trained model')
    saver.restore(model.sess, callback.model_dir)

    # load obs_rms
    import pickle
    with open(callback.directory + '/ob_rms.pkl', 'rb') as output:
        ob_rms = pickle.load(output)


    def filter(ob):
        ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10, 10)
        return ob

    obsdict = []
    acsdict = []
    assert len(env.venv.envs) == 1
    env_test = env.venv.envs[0]
    for e in range(500):
        obsrecord = []
        acsrecord = []
        done = False
        rsum = 0
        obs = env_test.reset()
        obs_orig = obs.copy()
        obsrecord.append(obs_orig)
        obs = filter(obs)
        #env_test.render()
        while not done:
            #print(obs.shape)
            obs = np.expand_dims(obs.flatten(), axis=0)
            actions, values, _, neglogpacs = model.step(obs)
            newobs, r, done, _ = env_test.step(actions)
            obs = newobs
            obs_orig = obs.copy()
            obsrecord.append(obs_orig)
            acsrecord.append(actions)
            obs = filter(obs)
            
        obsdict.append(np.array(obsrecord))
        acsdict.append(np.array(acsrecord))

    if callback is not None:
        callback(locals(), globals())
