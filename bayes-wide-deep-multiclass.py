#!/usr/bin/env python
# coding: utf-8

# # Machine learning for Transformer Health Index Analysis

# <p style="background-color:azure;padding:10px;border:2px solid lightsteelblue"><b>Author:</b> Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# <br>
# University of Split, FESB, Department of Power Engineering <br>R. Boskovica 32, HR-21000 Split, Croatia, EU.</p>

# In[1]:


from __future__ import print_function


# In[2]:


import warnings
import exceptions as ex


# In[3]:


# Ignore warnings
warnings.filterwarnings(action='ignore', category=ex.FutureWarning)


# In[4]:


from IPython.display import Image


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[6]:


from scipy import stats
from scipy import optimize
# Scikit-learn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


# In[7]:


import keras


# In[8]:


import theano
import theano.tensor as tt
from keras.wrappers.scikit_learn import KerasClassifier


# In[9]:


# Inline figures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


# Figure aesthetics
sns.set(context='notebook', style='white', font_scale=1.2)
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})


# Proposed overall model incorporates **joint** and **ensemble** learning, using combination of three different individual machine learning models. Joint learning itself combines **wide and deep** models in the Bayesian setting. The **wide model** comes from the Bayesian multinomial ordered "probit" regression (which is a multiclass classification model that preserves inherent ordering between categories without imposing distance measure between the categories). The **deep model** is a Bayesian feed-forward artifical neural network (with several hidden layers) and a softmax activation on the output layer, making it appropriate for the multiclass classification (without considering ordering between categories). Both model parts, wide and deep, are constructed in the Bayesian setting and are **jointly trained** using the Markov Chain Monte Carlo (MCMC) algorithm. Ensemble learning combines a Bayesian multinomial ordered probit regression with a deep neural network (densly connected multilayer ANN) using the weighted **soft voting** principle. The deep neural network architecture (which is composed of many densely connected, following by, regularizing layers) along with some of its hyper-parameters, is optimised using the grid search with cross-validation. The third individual model is itself an ensemble of tree-based machine learning models for the multiclass classification. The hyper-parameters of this model (ensemble of trees) is optimised using the grid search with cross-validation. The **final ensamble** of three different groups of models is carried out using the (equal weighted) **hard voting** principle.

# In[10]:


Image(filename="diagram.png", width=500)


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Ensembling consists of pooling together the predictions of a set of different models, to produce better predictions. The key to making ensembling work is the diversity of the set of classifiers. Diversity is what makes ensembling work. For this reason, one should ensemble models that are as good as possible while being <b>as different as possible</b>. This typically means using very different network architectures or even different brands of machine-learning approaches. This is exactly what has been proposed here.</p>

# ### Transformer diagnostic data and health index values

# Transformer diagnostic tests are described in terms of the following six parameters:
# * Water (ppm)
# * Acidity (mgKOH/g)
# * DBV (kV)
# * Dissipation factor (%)
# * TDCG (ppm)
# * Furan (mg/L)

# In[11]:


data = pd.read_csv('Health_index.csv')
data.drop(labels=['No.'], axis=1, inplace=True)
data


# In[12]:


data['GRNN'].value_counts()/data['GRNN'].value_counts().sum()


# In[13]:


# Correcting outlier with the median value of the data
median = np.percentile(data['DF'], q=50)
data.set_value(18, 'DF', median)
data.iloc[17:20]


# In[14]:


# Column names
values = ['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan']
values_all = ['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan', 'GRNN-S']


# In[15]:


warnings.filterwarnings(action='ignore', category=ex.UserWarning)


# In[16]:


g = sns.PairGrid(data[values], size=1.2)  # size in inches?!
g.map_upper(plt.scatter);
g.map_lower(sns.kdeplot, cmap='Blues_d');
g.map_diag(plt.hist);


# ### Pearson's correlation matrix (Predictivity)

# In[17]:


# Predictivity (Pearson correlation matrix)
pearson = data[values_all].corr('pearson')
# Correlation matrix as heatmap (seaborn)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(pearson, annot=True, annot_kws=dict(size=10), vmin=-1, vmax=1, cmap=plt.cm.coolwarm, ax=ax)
plt.tight_layout()
plt.show()


# ### Generate syntetic transformer data

# In[15]:


# Generate synthetic data using the "data augmentation" technique
def trafo_measurements(df, num=100, fraction=0.1):
    data = {}
    idxmax = len(df.index)
    ranvals = np.random.randint(low=0, high=idxmax, size=num)
    for name in df.columns:
        if name == 'GRNN-S':
            data[name] = df[name].iloc[ranvals]
        else:
            sd = df[name].std()
            datavals = df[name].iloc[ranvals].values
            ransigns = np.random.choice([-1., 1.], size=num, replace=True)
            synvalues = datavals + ransigns*(sd*fraction)
            values = np.empty_like(synvalues)
            for i, val in enumerate(synvalues):
                if val > 0. or val is not np.NaN:
                    values[i] = val
                else:
                    values[i] = datavals[i]
            data[name] = np.round(values, decimals=3)
    data = pd.DataFrame(data, columns=df.columns)
    return data


# Health Index values:
# VB = 4 -> Very bad
#  B = 3 -> Bad
#  M = 2 -> Moderate
#  G = 1 -> Good
# VG = 0 -> Very good 

# In[16]:


def from_descr_to_score(description):
    if description == 'VB':
        score = 4
    elif description == 'B':
        score = 3
    elif description == 'M':
        score = 2
    elif description == 'G':
        score = 1
    elif description == 'VG':
        score = 0
    else:
        score = None
        raise ValueError('Invalid class label {} encountered!'.format(description))
    return score

def from_score_to_descr(score):
    if score <= 0.2:
        description = 'VG'
    elif 0.2 < score <= 0.4:
        description = 'G'
    elif 0.4 < score <= 0.6:
        description = 'M'
    elif 0.6 < score <= 0.8:
        description = 'B'
    elif 0.8 < score:
        description = 'VB'
    else:
        description = None
        raise ValueError('Score value {} is outside the intended range [0-1]!'.format(score))
    return description

def from_class_to_label(cl):
    if cl == 4:
        label = 'VB'
    elif cl == 3:
        label = 'B'
    elif cl == 2:
        label = 'M'
    elif cl == 1:
        label = 'G'
    elif cl == 0:
        label = 'VG'
    else:
        label = None
        raise ValueError('Invalid class index {} encountered!'.format(cl))
    return label


# In[17]:


# Generate syntetic data using "data augmentation"
syntetic = trafo_measurements(data[values_all], num=100)
syntetic['GRNN'] = syntetic['GRNN-S'].apply(from_score_to_descr)


# In[18]:


data['score'] = data['GRNN'].apply(from_descr_to_score)
syntetic['score'] = syntetic['GRNN'].apply(from_descr_to_score)
syntetic.head()


# ### Data preprocessing and splitting

# In[19]:


# Hand-made data split 
x_train2 = syntetic[values]
x_test2  = data[values]


# In[19]:


# Standardize the data
def standardize(X_data):
    x_mean = X_data.mean(axis=0)
    x_std = X_data.std(axis=0)
    Z_data = ((X_data - x_mean)/x_std).values
    return Z_data


# In[20]:


Z_data = standardize(x_train2)


# In[21]:


y_train_log = syntetic['score'].values


# In[22]:


n_cat = syntetic['score'].unique().size  # No. categories


# ## Bayesian Multinomial Ordered Probit Regression

# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Multinomial ordered "probit" regression assumes a linear combination of the multiple metric predictors (without interaction) and an <b>ordered</b> categorical predicted variable, without imposing distance metric between the categories.</p>

# In[27]:


thresh = [k+0.5 for k in range(n_cat-1)]
thresh_obs = np.ma.asarray(thresh)
thresh_obs[1:-1] = np.ma.masked
print('thresh:\t{}'.format(thresh))
print('thresh_obs:\t{}'.format(thresh_obs))


# <p style="background-color:ivory;padding:10px;border:2px solid silver"><b>Note:</b> See the excellent book by J. K. Kruschke, Doing Bayesian Data Analysis: A tutorial with R, JAGS and Stan (2nd Edition), Academic Press, and the following <a href="https://github.com/JWarmenhoven/DBDA-python">github repository</a> for additional information.</p>

# In[28]:


from theano.compile.ops import as_op
# Theano cannot compute a gradient for these custom functions, so it is not possible to use
# gradient based samplers in PyMC3.
@as_op(itypes=[tt.dvector, tt.dvector, tt.dscalar], otypes=[tt.dmatrix])
def outcome_probabilities(theta, mu, sigma):
    out = np.empty((mu.size, n_cat))
    n = stats.norm(loc=mu, scale=sigma)  # Normal distribution
    out[:,0] = n.cdf(theta[0])        
    out[:,1] = np.max([np.repeat(0,mu.size), n.cdf(theta[1]) - n.cdf(theta[0])], axis=0)
    out[:,2] = np.max([np.repeat(0,mu.size), n.cdf(theta[2]) - n.cdf(theta[1])], axis=0)
    out[:,3] = np.max([np.repeat(0,mu.size), n.cdf(theta[3]) - n.cdf(theta[2])], axis=0)
    out[:,4] = 1. - n.cdf(theta[3])
    return out


# In[ ]:


# Version of the ordered "probit" regression that is robust against outliers
# It uses Student's t distribution instead of the Normal distribution
from theano.compile.ops import as_op
# Theano cannot compute a gradient for these custom functions, so it is not possible to use
# gradient based samplers in PyMC3.
@as_op(itypes=[tt.dvector, tt.dvector, tt.dscalar, tt.dscalar], otypes=[tt.dmatrix])
def outcome_probabilities_robust(theta, mu, sigma, nu):
    out = np.empty((mu.size, n_cat))
    n = stats.t(df=nu, loc=mu, scale=sigma)  # Student-T distribution
    out[:,0] = n.cdf(theta[0])        
    out[:,1] = np.max([np.repeat(0,mu.size), n.cdf(theta[1]) - n.cdf(theta[0])], axis=0)
    out[:,2] = np.max([np.repeat(0,mu.size), n.cdf(theta[2]) - n.cdf(theta[1])], axis=0)
    out[:,3] = np.max([np.repeat(0,mu.size), n.cdf(theta[3]) - n.cdf(theta[2])], axis=0)
    out[:,4] = 1. - n.cdf(theta[3])
    return out


# In[29]:


with pm.Model() as probit:
    # Priors
    theta = pm.Normal('theta', mu=thresh, tau=np.repeat(1./2**2, len(thresh)),
                      shape=len(thresh), observed=thresh_obs)
    zbeta0 = pm.Normal('zbeta0', mu=float(n_cat)/2., tau=1./n_cat**2)
    zbeta = pm.Normal('zbeta', mu=0., tau=1./n_cat**2, shape=Z_data.shape[1])
    zsigma = pm.Uniform('zsigma', n_cat/1000., n_cat*10.)
    # Linear model
    mu = pm.Deterministic('mu', zbeta0 + pm.math.dot(zbeta, Z_data.T))
    # Link function
    pr = outcome_probabilities(theta, mu, zsigma)
    # For the *robust* version of the ordered "probit" regression
    # comment the previous line and uncomment the following lines
    ##nu = pm.Exponential('nu', lam=1./30.)
    ##pr = outcome_probabilities_robust(theta, mu, zsigma, nu)
    # Likelihood
    y = pm.Categorical('y', pr, observed=y_train_log)
    # MCMC (it is not possible to use gradient-based samplers)
    step_M = pm.DEMetropolis()  # experimental sampler!
    chain = pm.sample(draws=32000, tune=4000, step=step_M, chains=4, parallelize=True)


# In[30]:


burnin = 2000
thin = 6
# Trace after burn-in and thinning
trace = chain[burnin::thin]


# In[31]:


pm.gelman_rubin(chain, varnames=['theta_missing', 'zbeta0', 'zbeta', 'zsigma'])


# In[32]:


pm.traceplot(trace, varnames=['theta_missing', 'zbeta0', 'zbeta', 'zsigma']);


# In[33]:


# Convert parameters to original scale
x_mean = x_train2.mean(axis=0).values
x_std  = x_train2.std(axis=0).values

beta = trace['zbeta']/x_std
beta0 = trace['zbeta0'] - np.sum(trace['zbeta']*x_mean/x_std, axis=1)
sigma = trace['zsigma']

# Concatenate the fixed thresholds into the estimated thresholds
n = trace['theta_missing'].shape[0]  # No. chain values
thresholds = np.c_[np.tile([0.5], (n,1)), trace['theta_missing'], np.tile([3.5], (n,1))]


# In[34]:


# Priors on the scale of the original data
zbeta0_prior = pm.Normal.dist(mu=float(n_cat)/2., tau=1./n_cat**2).random(size=10000)
zbeta_prior = pm.Normal.dist(mu=0., tau=1./n_cat**2, shape=6).random(10000)
beta_prior = zbeta_prior/x_std
beta0_prior = zbeta0_prior - np.sum(zbeta_prior*x_mean/x_std)


# In[35]:


fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(5,8))
ax[0,1].set_axis_off()
#ax[0,1].set_title('$\sigma$')
#pm.plot_posterior(sigma, point_estimate='mode', text_size=14, ax=ax[0,1]);
pm.plot_posterior(beta0, point_estimate='mode', text_size=14, ax=ax[0,0]);
ax[0,0].set_title('Interc.')
k = 0
for i in range(1,4):
    for j in range(2):
        k += 1
        pm.plot_posterior(beta[:,k-1], point_estimate='mode', text_size=14, ax=ax[i,j]);
        #pm.kdeplot(beta_prior, color='brick', linestyle='--', lw=1.5, ax = ax[i,j])
        ax[i,j].set_title(values[k-1])
plt.tight_layout()
plt.show()


# In[36]:


# Posterior prediction from chain 
def posterior_predictive_from_chain(n, thresholds, beta0, beta, X, mu, sigma):
    out = []
    for i in range(n):
        mu = beta0[i] + np.dot(X, beta[i,:])
        # Normal distribution
        threshCumProb = stats.norm().cdf((thresholds[i,:] - mu)/sigma[i])
        outProb = np.empty(len(threshCumProb)+1)
        for k in range(1, thresholds.shape[1]):
            outProb[k] = threshCumProb[k] - threshCumProb[k-1]
        outProb[0] = threshCumProb[0]
        outProb[thresholds.shape[1]] = 1. - threshCumProb[-1]
        out.append(outProb)
    out = np.asarray(out)
    return out


# In[37]:


# Posterior prediction from chain for the robust model
def posterior_predictive_robust_from_chain(n, thresholds, beta0, beta, X, mu, sigma, nu):
    out = []
    for i in range(n):
        mu = beta0[i] + np.dot(X, beta[i,:])
        # Student's t distribution
        threshCumProb = stats.t(df=nu[i]).cdf((thresholds[i,:] - mu)/sigma[i])
        outProb = np.empty(len(threshCumProb)+1)
        for k in range(1, thresholds.shape[1]):
            outProb[k] = threshCumProb[k] - threshCumProb[k-1]
        outProb[0] = threshCumProb[0]
        outProb[thresholds.shape[1]] = 1. - threshCumProb[-1]
        out.append(outProb)
    out = np.asarray(out)
    return out


# In[38]:


# Example of the posterior prediction with uncertainty of estimation
posterior_proba = posterior_predictive_from_chain(n, thresholds, beta0, beta, x_test2.values[3], mu, sigma)
posterior = pd.DataFrame(posterior_proba, columns=['VG', 'G', 'M', 'B', 'VB'])
fig, ax = plt.subplots(figsize=(5,3.5))
sns.boxplot(data=posterior, palette='Set2', linewidth=1.5, ax=ax, showmeans=True)
ax.grid(axis='y')
ax.set_xlabel('Health Index')
ax.set_ylabel('Probability')
plt.tight_layout()
plt.show()


# In[39]:


# Predict class probability on test data
proba = []
for X in x_test2.values:
    out = posterior_predictive_from_chain(n, thresholds, beta0, beta, X, mu, sigma)
    proba.append(out.mean(axis=0))
proba = np.asarray(proba)
argmaxs = proba.argmax(axis=1)
labels_pred = map(from_class_to_label, argmaxs)


# In[20]:


y_t = data[['GRNN']].copy()  # instantiate dataframe for comparisons


# In[41]:


#y_t['b'] = argmaxs
y_t['bayes'] = labels_pred


# ## Wide & Deep Bayesian multiclass classification model

# The **wide part** of the model consists of the **Bayesian multinomial ordered "probit" regression**, which is a multiclass classification model that preserves inherent oredering between the categories, without imposing distance metric between the categories. The wide part of the model captures linear relationships between the predictors and predicted variable (including any interactions between predictors), as is the case with generalized linear models. The **deep part** of the model is a **feed-forward artifical neural network** (ANN) with arbitrary number of hidden layers (where the number of neurons per layer can be adjusted, along with its activation functions), which means that the network can be **deep** with many densely connected layers. The last layer uses **"softmax"** activation, which turns the network into the **multiclass classification model** (without inherent odreding between the categories). The deep part of the model can capture any non-linear relationship between the predictors and predicted variable. The outputs of the wide and deep parts, which are the class outcome probabilities, are combined using the **weighted avarage** and fed into the likelihood function (which is the Categorical distribution). The complete model, which consists of the wide and deep parts, is jointly trained using the **Markov Chain Monte Carlo** (MCMC) algorithm.

# In[42]:


Image(filename="kruschke_style.png", width=500)


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> The wide part of the Bayesian model can be easily adapted to, and could probably benefit from, using data structure with multiple groups (e.g. distribution and transmission level transformers). This would allow for the more efficient use of unsymmetrical datasets, and would enable extending the present wide part of the model with the additional <b>hierarchical multilevel structure</b> that would allow informing model parameters across different groups through the mutual higher level hyper-priors.</p>

# In[21]:


# Train data (synthetic)
X_train = syntetic[values]
y_train = syntetic['score'].values
# Test data (actual)
X_test = data[values]
y_test = data['score'].values


# In[22]:


# Standardize the input data for the neural network
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Feed-forward artifical neural network (ANN) can consist of the arbitrary number of layers (where different layers can have the same or different number of neurons) and different activation functions. Each layer is implemented as: $\sigma(W \cdot x + b)$ where $\sigma$ is the activation function, $W$ is the weights matrix and $b$ is the bias vector. In implementing the network architecture, user needs to keep track of the dimensions of the weight matrices and bias vectors for the different layers. In addition, dropout layers are also implemented in order to introduce network regularization that prevents it from the overfitting.</p>

# In[43]:


floatX = theano.config.floatX


# In[46]:


def construct_wide_and_deep_model(ann_input, ann_output):
    """
    Network (deep part) can be extended to account for additional 
    layers (or use different activation functions on hidden layers)
    """
    # Number of neurons per layer (ANN)
    n_hidden = 10  # input
    # Note: Network can have different number of neurons per layer,
    # in which case these would be defined here. The shapes of the 
    # tensors for different layers would need to reflect this fact,
    # which is simple matter of index house-keeping for layers. The
    # same goes for the layer random initializations.
    
    # Fixed weight for model joining
    #weight = 0.5  # equal-weighted
    # Note: With defining a small "weight" value, larger emphasis 
    # is given to the deep model part (ANN), and vice-versa. As an
    # alternative, weight can be left to be estimated by the model
    # itself, with an informed prior on the scale of [0,1]:
    # weight = pm.Beta('weight', alpha=10, beta=10)

    # Initialize random weights for layers
    init_bias = np.random.randn(n_hidden).astype(floatX)
    init_bias_out = np.random.randn(n_cat).astype(floatX)
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden, n_cat).astype(floatX)

    with pm.Model() as model:
        # --------------------------------------------
        # Multinomial "probit" regression (wide model)
        # --------------------------------------------
        # Priors
        theta = pm.Normal('theta', mu=thresh, tau=np.repeat(1./2**2, len(thresh)),
                          shape=len(thresh), observed=thresh_obs)
        zbeta0 = pm.Normal('zbeta0', mu=float(n_cat)/2., tau=1./n_cat**2)
        zbeta = pm.Normal('zbeta', mu=0., tau=1./n_cat**2, shape=X_train.shape[1])
        #zsigma = pm.Uniform('zsigma', n_cat/1000., n_cat*10.)
        zsigma = pm.HalfCauchy('zsigma', beta=float(n_cat)/2.)  # alternative prior
        # Linear model part
        mu = pm.Deterministic('mu', zbeta0 + pm.math.dot(ann_input, zbeta))
        # Outcome probabilities (see outside function)
        pr = outcome_probabilities(theta, mu, zsigma)
        # --------------------------------------------
        # Feed-forward neural network (deep model)
        # --------------------------------------------
        std = float(n_cat)
        # Bias of the input layer
        b1 = pm.Normal('b1', mu=0., sd=std, shape=n_hidden, testval=init_bias)
        # Weights from input to hidden layer
        w1 = pm.Normal('w1', mu=0., sd=std, shape=(X_train.shape[1], n_hidden), 
                        testval=init_1)
        # Bias of the hidden layer
        b2 = pm.Normal('b2', mu=0., sd=std, shape=n_hidden, testval=init_bias)
        # Weights from 1st to 2nd layer
        w12 = pm.Normal('w12', mu=0., sd=std, shape=(n_hidden, n_hidden), 
                         testval=init_2)
        # Dropout: regularization
        dropout = pm.Bernoulli('dropout', p=0.25, shape=n_hidden)  # input: p
        # *** Add additional hidden layers here ***
        # Bias of the output layer
        b3 = pm.Normal('b3', mu=0., sd=std, shape=n_cat, testval=init_bias_out)
        # Weights from hidden layer to output
        w3 = pm.Normal('w3', mu=0., sd=std, shape=(n_hidden, n_cat), 
                        testval=init_out)
        # Build feed-forward neural network from layers (additional layers can be added)
        z1 = pm.math.dot(ann_input, w1) + b1
        act_1 = pm.math.tanh(z1)  # or tt.tanh(z1) -> "tanh" activation function
        # Other activation functions could be used, such as:
        #act_1 = tt.nnet.relu(z1)  # "relu" activation function
        #act_1 = tt.nnet.sigmoid(z1)  # "sigmoid" activation func.
        z2 = pm.math.dot(act_1, w12) + b2
        act_2 = pm.math.tanh(z2)
        # Introduce dropout layer
        act_2 = dropout * act_2
        # *** Process additional hidden layers here ***
        z3 = pm.math.dot(act_2, w3) + b3
        # Softmax activation for outcome probabilities
        act_out = tt.nnet.softmax(z3)
        # --------------------------------------------
        # Joining wide & deep model parts 
        # --------------------------------------------
        # Define informed prior on weight parameter
        weight = pm.Beta('weight', alpha=10, beta=10)
        # Weighted average of probabilities from wide & deep parts
        proba = pm.Deterministic('proba', pr*weight + act_out*(1. - weight))

        # Multiclass classification -> Categorical likelihood
        out = pm.Categorical('out', p=proba, observed=ann_output)
    return model


# In[47]:


ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)
model = construct_wide_and_deep_model(ann_input, ann_output)


# In[48]:


# MCMC sampling (it is not possible to use gradient-based samplers)
with model:
    step_M = pm.Metropolis()
    chain = pm.sample(draws=32000, tune=6000, step=step_M, chains=1, parallelize=True)


# In[49]:


# Trace after burn-in and thinning
model_trace = chain[burnin::thin]


# In[50]:


pm.traceplot(model_trace, varnames=['theta_missing', 'zbeta0', 'zbeta', 'zsigma']);


# In[51]:


# Predict class probability on test data
ann_input.set_value(X_test)
ann_output.set_value(y_test)
with model:
    model_ppc = pm.sample_ppc(model_trace, samples=1000)


# In[52]:


y_t['m'] = np.median(model_ppc['out'], axis=0)
y_t['model'] = y_t['m'].apply(from_class_to_label)
del y_t['m']


# ## Keras deep neural network

# In[30]:


# Encode 'targets' to categorical variables
y_train_nn = keras.utils.np_utils.to_categorical(y_train)
y_test_nn = keras.utils.np_utils.to_categorical(y_test)


# In[31]:


BATCH = 32     # 'batch_size' argument
EPOCHS = 200   # 'epochs' argument


# #### Keras DNN with grid search and cross-validation for network architecture and hyper-parameters optimisation

# In[55]:


def keras_dnn(hidden_layers, number_neurons, activation, dropout):
    # Deep feed-forward artificial neural network
    model = keras.models.Sequential()
    # Input layer
    model.add(keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
    # Hidden layers architecture
    for layer in range(hidden_layers):
        # Add hidden layer (densely connected)
        model.add(keras.layers.Dense(units=number_neurons, activation=activation))
        # Add regularizing layer
        model.add(keras.layers.Dropout(dropout))
    # Output layer
    model.add(keras.layers.Dense(units=5, activation='softmax'))  # five classes
    # Optimizer
    adam = keras.optimizers.Adam()
    # Compile network
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# In[56]:


# Instantiate Keras Classifier scikit-learn API (wrapped keras model)
network = KerasClassifier(build_fn=keras_dnn, epochs=100, batch_size=BATCH)
# Instantiate GridSearchCV
dnn = GridSearchCV(network, param_grid={'hidden_layers':[3, 4],
                                        'number_neurons':[64, 32],
                                        'activation':['relu', 'tanh'], 
                                        'dropout':[0.5]},
                   scoring='neg_log_loss', cv=3, n_jobs=1, verbose=0)
# Performing grid search with a large number of parameters 
# can be very expensive from the execution time standpoint
dnn.fit(X_train, y_train_nn, verbose=0)


# In[57]:


print('The parameters of the best DNN are: ')
print(dnn.best_params_)


# In[58]:


# Use the best DNN for prediction
best_dnn = dnn.best_estimator_.model  # unwrapped keras model
y_test_eval_proba = best_dnn.predict_proba(X_test)


# In[59]:


# Network architecture
best_dnn.summary()


# In[60]:


# Score metrics (evaluate best model on test data)
score_best = best_dnn.evaluate(X_test, y_test_nn, batch_size=BATCH, verbose=0)
print('Log-loss: {:g}, Accuracy: {:.2f} %'.format(score_best[0], score_best[1]*100))


# In[61]:


y_t['b'] = y_test_eval_proba.argmax(axis=1)
y_t['best_dnn'] = y_t['b'].apply(from_class_to_label)
del y_t['b']


# #### Keras DNN for "probit" regression with fixed number of layers and hyper-parameters

# In[32]:


from tensorflow.distributions import Normal as TFNormal

def probit(x):
    normal = TFNormal(loc=0., scale=1.)
    return normal.cdf(x)

# Keras: Feed-forward neural network for ordinal probit regression
model = keras.models.Sequential()
# Input layer
model.add(keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(keras.layers.Dropout(0.5))  # regularization layer
# hidden layer
model.add(keras.layers.Dense(units=64, activation='tanh'))
model.add(keras.layers.Dropout(0.5))
# hidden layer
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dropout(0.25))
# hidden layer
model.add(keras.layers.Dense(units=32, activation='tanh'))
model.add(keras.layers.Dropout(0.25))
# *** Add additional hidden layers here ***
# Output layer
model.add(keras.layers.Dense(units=5, activation=probit))
# Optimizer
adam = keras.optimizers.Adam()
# Compile network
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[33]:


# Fit model on train data
history = model.fit(X_train, y_train_nn, epochs=EPOCHS, batch_size=BATCH, 
                    validation_data=(X_test, y_test_nn), shuffle=True, verbose=0)
# Score metrics (evaluate model on test data)
score = model.evaluate(X_test, y_test_nn, batch_size=BATCH, verbose=0)
print('Log-loss: {:g}, Accuracy: {:.2f} %'.format(score[0], score[1]*100))


# In[34]:


hist = history.history
acc = hist['acc']
acc_val = hist['val_acc']
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(acc, ls='-', lw=2, c='seagreen', label='test accuracy')
ax.plot(acc_val, ls='-', lw=2, c='royalblue', label='validation accuracy')
ax.grid()
ax.legend(loc='best', fontsize=12)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.show()


# In[35]:


# Predict class probability on test data
y_pred_proba_nn = model.predict_proba(X_test, batch_size=BATCH, verbose=0)
y_t['k'] = y_pred_proba_nn.argmax(axis=1)
y_t['keras'] = y_t['k'].apply(from_class_to_label)
del y_t['k']


# ## Ensemble models using "soft voting"

# #### Optimize model weights

# In[89]:


# individual model predictions (Bayes & ANN)
predictions = [proba, y_test_eval_proba]


# In[90]:


# loss function for the optimization
def loss_function(weights):
    final_prediction = 0.
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    # using scikit-learn loss for the classification
    loss_value = metrics.log_loss(y_test, final_prediction)
    return loss_value


# In[91]:


# https://www.kaggle.com/hsperr/finding-ensamble-weights
# algorithm needs a starting value
start_vals = [1./len(predictions)]*len(predictions)
# add constraints
cons = ({'type':'eq','fun':lambda w: 1. - np.sum(w)})
# weights are bound between 0 and 1
buns = [(0., 1.)]*len(predictions)
# minimize loss function: scipy.optimize.minimize 
# using SLSQP method with constraints
res = optimize.minimize(loss_function, start_vals, method='SLSQP', 
                        bounds=buns, constraints=cons)
weights = res['x']  # ensemble model's weights
print('Optimal weights: {}; Sum of weights: {}'.format(weights.round(3), weights.sum()))


# #### Alternative: Use fixed weights

# In[ ]:


weights = np.asarray([0.5, 0.5])  # sum to one


# #### Ensemble models using soft voting with weights

# In[92]:


# Models: Bayes & ANN
proba_weighted = np.empty_like(proba)
for k in range(proba.shape[0]):
    predictions = np.vstack((proba[k], y_test_eval_proba[k])).T  
    proba_weighted[k] = np.dot(predictions, weights)


# In[93]:


y_t['e'] = proba_weighted.argmax(axis=1)
y_t['soft_vote'] = y_t['e'].apply(from_class_to_label)
del y_t['e']


# ## ExtraTreesClassifier

# In[94]:


# ExtraTreesClassifier (ensemble learner) with grid search 
# and cross-validation for hyper-parameters optimisation
parameters = {'n_estimators':[5, 10, 15, 20], 
              'criterion':['gini', 'entropy'], 
              'max_depth':[2, 5, None]}
trees = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=parameters, 
                     cv=3, scoring='neg_log_loss', refit=True, n_jobs=-1) 
trees.fit(X_train, y_train)
y_trees = trees.predict_proba(X_test)


# In[95]:


# Best model parameters
trees.best_params_


# In[96]:


y_t['t'] = y_trees.argmax(axis=1)
y_t['trees'] = y_t['t'].apply(from_class_to_label)
del y_t['t']


# ## Ensemble models using "hard voting"

# In[97]:


# Models: Wide&Deep & Ensemble & ExtraTrees
# Each model has equal weight. 
# In general, a simple weighted average with weights optimized
# on the validation data could provide a very strong baseline.
hard_vote = []
wide_deep = np.median(model_ppc['out'], axis=0)
for k in range(proba.shape[0]):
    hard_vote.append((wide_deep[k].argmax(), proba_weighted[k].argmax(), y_trees[k].argmax()))
counts = [np.bincount(x).argmax() for x in hard_vote]
y_t['h'] = counts
y_t['hard_vote'] = y_t['h'].apply(from_class_to_label)
del y_t['h']


# #### Predictions using individual classifiers and ensembles

# In[98]:


y_t


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Reported model accuracy depends on the random synthetic dataset used during the learning phase, which has been generated from the original dataset (used for testing) by means of the simple "data augmentation" technique. Possibility for overfitting and underfitting should be further examined, preferably with a larger dataset.</p>

# ## Supplementary information and models

# #### Deep neural network with multiple outputs (heads)

# Following artificial neural network, implemented in **Keras**, is attempting to produce a **regression and a multiclass classification predictions at the same time**, using the double output (i.e. using two "heads"). Training such a model requires the ability to specify different loss functions for different heads of the network: regression task uses the loss function "mse", while the multiclass classification task uses the loss function "categorical_crossentropy" (see the Keras documentation at keras.io). Model needs to be implemented using the Keras functional API appraoch.

# In[31]:


# Train & test data for the regression head
y_train_reg = syntetic['GRNN-S'].values
y_test_reg = data['GRNN-S'].values


# In[49]:


# Input layer
inputs = keras.Input(shape=(X_train.shape[1],))
x = keras.layers.Dense(64, activation='relu')(inputs)  # Dense layer
x = keras.layers.Dropout(0.5)(x)  # Dropout layer
x = keras.layers.Dense(64, activation='tanh')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Dense(32, activation='tanh')(x)
x = keras.layers.Dropout(0.25)(x)
# *** Add additional layers here ***
# ------------------------------
# Multiple heads
# ------------------------------
# Multiclass classification head
output_klas = keras.layers.Dense(5, activation='softmax', name='classifier')(x)
# Regression head
output_regr = keras.layers.Dense(1, name='regressor')(x)

# Instantiate Model
model = keras.models.Model(inputs, [output_klas, output_regr])
# Compile the model
#opt = keras.optimizers.RMSprop(lr=0.0004, decay=1e-8)  # optimizer parameters
model.compile(optimizer='rmsprop', loss={'classifier': 'categorical_crossentropy', 
                                         'regressor': 'mse'}, 
              loss_weights={'classifier': 2., 'regressor': 1.})  # loss function weights


# In[50]:


# Train model on test data
history = model.fit(X_train, {'classifier': y_train_nn, 
                              'regressor': y_train_reg}, 
                    epochs=EPOCHS, batch_size=BATCH, verbose=0)


# In[51]:


# Predict on test data
y_pred_klas, y_pred_reg = model.predict(X_test, batch_size=BATCH, verbose=1)


# In[52]:


y_2 = data[['GRNN', 'GRNN-S']].copy()  # instantiate dataframe for comparisons
y_2['k'] = y_pred_klas.argmax(axis=1)
y_2['class'] = y_2['k'].apply(from_class_to_label)
del y_2['k']
y_2['regrs'] = y_pred_reg
y_2


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> It can be seen that the regression values can be negative and, hence, sometimes extend beyond the intended range of the health index space (which here assumes a range [0-1]).</p>

# #### A random forest multiclass classifier (ensemble learner)

# In[ ]:


# RandomForestClassifier (another ensemble learner for the multiclass 
# classification which can be used instead of the ExtraTreesClassifier)
parameters = {'n_estimators':[10, 15, 20], 
              'criterion':['gini', 'entropy'],
              'max_features':[4, 'auto'],
              'max_depth':[2, None]}
# grid search and cross-validation for hyper-parameters optimisation
forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, 
                      cv=3, scoring='neg_log_loss', refit=True, n_jobs=-1) 
forest.fit(X_train, y_train)
y_forest = forest.predict_proba(X_test)


# In[ ]:


y_t['f'] = y_forest.argmax(axis=1)
y_t['forest'] = y_t['f'].apply(from_class_to_label)
del y_t['f']


# #### Feature importance analysis with GradientBoosting classifier

# In[31]:


# Train & evaluate model performance
def train_and_evaluate(model, X, y, ns=3):
    # k-fold cross validation iterator 
    cv = KFold(n_splits=ns, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv)
    print('Score using {:d}-fold CV: {:g} +/- {:g}'.format(ns, np.mean(scores), np.std(scores)))


# In[37]:


# Data standardization
X_stand = preprocessing.scale(x_train2)
# Gradient Boosting Classifier
clf_gb = GradientBoostingClassifier()
train_and_evaluate(clf_gb, X_stand, y_train_log)
clf_gb.fit(X_stand, y_train_log)


# In[38]:


# Feature importance
feature_importance = clf_gb.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
# Plot relative feature importance
fig, ax = plt.subplots(figsize=(6,4))
ax.barh(pos, feature_importance[sorted_idx], align='center', color='seagreen', alpha=0.6)
plt.yticks(pos, data.columns[sorted_idx])
ax.set_xlabel('Feature Relative Importance')
plt.tight_layout()
plt.show()


# #### Bayesian Optimization with cross-validation for optimal model hyper-parameters

# In[ ]:


try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import Colours
    bayesian_optimisation = 1
except ImportError:
    print('Install Bayesian Optimisation package before executing the next cell!')
    bayesian_optimisation = 0


# In[ ]:


# Bayesian Optimization with cross-validation for the optimal model 
# hyper-parameters selection of the RandomForestClassifier
if bayesian_optimisation != 0:
    def optimize_RFC(data, targets):
    """Apply Bayesian Optimization to RandomForestClassifier hyper-parameters."""
    def RFC_CV(n_estimators, min_samples_split, max_features, data, targets):
        """RandomForestClassifier with cross validation."""
        estimator = RandomForestClassifier(n_estimators=n_estimators, 
                                           min_samples_split=min_samples_split, 
                                           max_features=max_features)
        cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss', cv=3)
        return cval.mean()

    def RFC_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForestClassifier cross validation."""
        return RFC_CV(n_estimators=int(n_estimators),
                      min_samples_split=int(min_samples_split),
                      max_features=max(min(max_features, 0.999), 1e-3),
                      data=data,
                      targets=targets)

    optimizer = BayesianOptimization(f=RFC_crossval, 
                                     pbounds={"n_estimators": (10, 25),
                                              "min_samples_split": (2, 25),
                                              "max_features": (0.1, 0.999)}, 
                                     verbose=2)
    optimizer.maximize(n_iter=10)
    return optimizer.max  # dictionary 

    print(Colours.green("--- Optimizing RandomForestClassifier ---"))
    optimal_params = optimize_RFC(X_train, y_train)
    print(optimal_params)


# #### Bayesian multinomial logistic regression (nominal predicted variable)

# In[41]:


with pm.Model() as model:
    # priors for categories 1-4, excluding reference category 0 which is set to zero below
    zbeta0_temp = pm.Normal('zbeta0_temp', mu=0., tau=1./20**2, shape=4)
    # using all (six) of the predictor variables
    zbetak_temp = pm.Normal('zbetak_temp', mu=0., tau=1./20**2, shape=(6, 4))
    # add prior values zero (intercept, predictors) for reference category 0
    zbeta0 = pm.Deterministic('zbeta0', tt.concatenate([[0], zbeta0_temp]))
    zbetak = pm.Deterministic('zbetak', tt.concatenate([tt.zeros((6, 1)), zbetak_temp], axis=1))
    # Multiple regression 
    mu = pm.Deterministic('mu', zbeta0 + pm.math.dot(Z_data, zbetak))
    # Theano softmax function (logistic)
    p = pm.Deterministic('p', tt.nnet.softmax(mu))
    # Likelihood
    y = pm.Categorical('y_hat', p=p, observed=y_train_log)
    # MCMC
    chain = pm.sample(draws=22000, tune=4000, chains=2, discard_tuned_samples=True)


# In[42]:


burnin = 2000
thin = 4
trace = chain[burnin::thin]


# In[43]:


pm.traceplot(chain, varnames=['zbeta0_temp', 'zbetak_temp']);


# In[44]:


# Transform model parameters back to the original scale
zbeta0 = trace['zbeta0']
zbetak = trace['zbetak']
x_mean = x_train2.mean(axis=0)
x_std  = x_train2.std(axis=0)
beta0 = zbeta0 - np.sum(zbetak*(np.tile(x_mean, (n_cat,1))/np.tile(x_std, (n_cat,1))).T, axis=1)
betak = np.divide(zbetak, np.tile(x_std, (n_cat,1)).T)
estimates = np.insert(betak, 0, beta0, axis=1)


# In[51]:


beta_values = ['Interc.'] + values
# Plot the posterior distributions
fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(13,13))
for (r,c), ax in np.ndenumerate(axes):
    pm.plot_posterior(estimates[:,r,c], point_estimate='mean', text_size=12, ax=ax);
# Setting labels for the outcomes
for ax, title in zip(axes[0,:], ['Very Good', 'Good', 'Moderate', 'Bad', 'Very Bad']):
    ax.set_title(title)
# Setting labels for the predictors
for ax, title in zip(axes[:,0], beta_values):
    ax.set_ylabel(title);


# In[52]:


# Predictions from softmax regression (using values from the MCMC chain)
def predict_nominal_from_chain(beta_0, beta_k, X):
    Y_pred = np.empty_like(beta_0)
    p = np.empty_like(beta_0)
    # For each value in chain
    for i in range(beta_0.shape[0]):
        Y_pred[i] = beta_0[i] + np.dot(X, beta_k[i])
        for j in range(beta_0.shape[1]):
            p[i,j] = np.exp(Y_pred[i,j])/np.sum(np.exp(Y_pred[i]))
    mean_proba = p.mean(axis=0)
    return mean_proba


# In[ ]:


# Predict class on new data
klasa = []
for k in range(x_test2.shape[0]):
    # Class probabilities
    proba = predict_nominal_from_chain(beta0, betak, x_test2.iloc[k].values)
    # Class labels
    klasa.append(from_class_to_label(np.argmax(proba)))
y_t['LogReg'] = np.asarray(klasa)


# #### Alternative implementation of the Bayesian multinomial ordered regression (ordinal predicted variable)

# <p style="background-color:ivory;padding:10px;border:2px solid silver"><b>Note:</b> See the excellent book by R. McElreath, Statistical Rethinking: A Bayesian course with examples in R and Stan, CRC Press, and a following 
# <a href="https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3">github repository</a> for additional information.</p>

# In[24]:


class Ordered(pm.distributions.transforms.ElemwiseTransform):
    name = "ordered"

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out
    
    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])


# In[25]:


class OrderedLogistic(pm.distributions.Categorical):
    """
    Ordered Logistic Categorical log-likelihood.
    Useful for regression on ordinal data values whose values range
    from 1 to K as a function of some predictor, :math:`\eta`. The
    cutpoints, :math:`c`, separate which ranges of :math:`\eta` are
    mapped to which of the K observed dependent variables.  The number
    of cutpoints is K - 1.  It is recommended that the cutpoints are
    constrained to be ordered.
    
    Parameters
    ----------
    eta : float
        The predictor.
    cutpoints : array
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges.  Do not explicitly set the first and last elements of
        :math:`cutpoints` to negative and positive infinity.
    """

    def __init__(self, eta, cutpoints, *args, **kwargs):
        self.eta = tt.as_tensor_variable(eta)
        self.cutpoints = tt.as_tensor_variable(cutpoints)
        # Sigmoid inverse link function
        pa = pm.math.sigmoid(tt.shape_padleft(self.cutpoints) - tt.shape_padright(self.eta))
        p_cum = tt.concatenate([tt.zeros_like(tt.shape_padright(pa[:, 0])),
                pa, tt.ones_like(tt.shape_padright(pa[:, 0]))], axis=1)
        proba = p_cum[:, 1:] - p_cum[:, :-1]
        super(OrderedLogistic, self).__init__(p=proba, *args, **kwargs)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name_eta = get_variable_name(dist.eta)
        name_cutpoints = get_variable_name(dist.cutpoints)
        return (r'${} \sim \text{{OrderedLogistic}}'
                r'(\mathit{{eta}}={}, \mathit{{cutpoints}}={}$'
                .format(name, name_eta, name_cutpoints))


# In[26]:


model_input = theano.shared(X_train)
model_output = theano.shared(y_train)


# In[40]:


with pm.Model() as ordered:
    # Priors
    theta = pm.Normal('theta', mu=thresh, tau=np.repeat(1./2**2, len(thresh)), 
                      shape=len(thresh), transform=Ordered())
    zbeta = pm.Normal('zbeta', mu=0., tau=1./n_cat**2, shape=6)  # all six predictors
    # Linear model
    mu = pm.Deterministic('mu', pm.math.dot(model_input, zbeta))
    # Likelihood
    y = OrderedLogistic('y', cutpoints=theta, eta=mu, observed=model_output)
    # MCMC
    chain = pm.sample(draws=32000, tune=6000, chains=2, nuts_kwargs={'target_accept':0.95}, 
                      discard_tuned_samples=True)


# In[41]:


burnin = 2000
thin = 6
# Trace after burn-in and thinning
trace = chain[burnin::thin]


# In[42]:


pm.traceplot(trace, varnames=['theta', 'zbeta']);


# In[43]:


# Predict class probability on test data
model_input.set_value(X_test)
model_output.set_value(y_test)
with ordered:
    model_ppc = pm.sample_ppc(trace, samples=1000)


# In[44]:


# Median point estimate for class labels
y_t['b2'] = np.median(model_ppc['y'], axis=0)
y_t['Bayes2'] = y_t['b2'].apply(from_class_to_label)
del y_t['b2']


# ### Bayesian ordinal regression with alternative link function

# **Note:** See the excellent book by Peter Congdon, "Applied Bayesian Modelling", Second Edition, Wiley, 2014.

# In[23]:


model_input = theano.shared(X_train)
model_output = theano.shared(y_train)


# In[29]:


with pm.Model() as model:
    # priors for five categories
    zbeta0 = pm.Normal('zbeta0', mu=0., tau=1./20**2, shape=5)
    # using all (six) predictor variables with five categories
    zbetak = pm.Normal('zbetak', mu=0., tau=1./20**2, shape=(6, 5))
    # Multiple regression 
    mu = pm.Deterministic('mu', zbeta0 + pm.math.dot(model_input, zbetak))
    # Inverse link function
    lam = pm.Normal('lambda', mu=0., tau=1./9.)
    # weights
    w1 = pm.Deterministic('w1', pm.math.exp(-pm.math.exp(3.5*lam + 2.)))
    w3 = pm.Deterministic('w3', pm.math.exp(-pm.math.exp(-3.5*lam + 2.)))
    w2 = pm.Deterministic('w2', 1. - w1 - w3)
    # left skewed extreme value (LSEV)
    h1 = pm.Deterministic('h1', 1. - pm.math.exp(-pm.math.exp(mu)))
    # logistic 
    h2 = pm.Deterministic('h2', pm.math.exp(mu)/(1. + pm.math.exp(mu)))
    # right skewed extreme value (RSEV)
    h3 = pm.Deterministic('h3', pm.math.exp(-pm.math.exp(-mu)))
    h = pm.Deterministic('h', w1*h1 + w2*h2 + w3*h3) 
    # Likelihood
    y = pm.Categorical('y_hat', p=h, observed=model_output)
    # MCMC
    algo = pm.Metropolis()
    chain = pm.sample(draws=85000, tune=10000, step=algo, chains=1, 
                      discard_tuned_samples=True)


# In[30]:


burnin = 5000
thin = 8
# Trace after burn-in and thinning
trace = chain[burnin::thin]


# In[31]:


pm.traceplot(trace, varnames=['zbeta0', 'zbetak', 'lambda']);


# Notice: Negative values of λ are obtained when the LSEV form is preferred, and positive values when the RSEV is
# preferred; λ = 0 corresponds to the logit link (i.e. logistic inverse link).

# In[32]:


# Predict class probability on test data
model_input.set_value(X_test)
model_output.set_value(y_test)
with model:
    model_ppc = pm.sample_ppc(trace, samples=1000)


# In[33]:


# Median point estimate for class labels
y_t['b3'] = np.median(model_ppc['y_hat'], axis=0)
y_t['Bayes3'] = y_t['b3'].apply(from_class_to_label)
del y_t['b3']


# In[58]:


import sys, IPython, platform, sklearn
print("Notebook createad on {:s} computer running {:s} and using:      \nPython {:s}\nIPython {:s}\nPyMC3 {:s}\nScikit-learn {:s}\nPandas {:s}\nNumpy {:s}\n"      .format(platform.machine(), ' '.join(platform.linux_distribution()[:2]), sys.version[:5], 
              IPython.__version__, pm.__version__, sklearn.__version__, 
              pd.__version__, np.__version__))

