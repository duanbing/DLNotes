from mxnet import autograd, nd

def f(a):
    b = a * 2;
    while b.norm().asscalar() < 1000:
        b = b * 2;
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = nd.random.normal(scale = 1);
a.attach_grad();
with autograd.record():
    c = f(a)
c.backward()
assert a.grad == c / a

