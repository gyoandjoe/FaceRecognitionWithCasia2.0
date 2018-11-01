import theano.tensor as T
import numpy as np
import theano

#random_droput = np.random.RandomState()
random_droput = np.random.RandomState()
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

fn = rng_droput.binomial(n=1,size=(5,5),p=0.9, dtype=theano.config.floatX)
#bajo = muchos ceros
#alto = muchos unos
tf = theano.function(
            inputs= [],
            outputs= fn,
            givens={}
        )
res = tf()
fn = rng_droput.binomial(n=1,size=(5,5),p=0.9, dtype=theano.config.floatX)
res2 = tf()
print ("ok")