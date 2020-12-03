The docker image contains the compiled examples from the `examples` folder and
may be used to test various solver options with matrices in the MatrixMarket
format.

Run solver from the docker image:

```
$ docker run -ti -v $PWD:/data amgcl/examples solver -A A.mtx -f b.mtx
```
