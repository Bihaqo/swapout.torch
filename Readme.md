This is my Torch reimplementation of the paper _Swapout: Learning an ensemble of deep architectures_ ([arXiv](https://arxiv.org/abs/1605.06465)).

It can be used as `nn.CAddTable`:
```
nn.Sequential()
:add(nn.ConcatTable()
  :add(input)
  :add(f_input))
:add(nn.Swapout({0.6, 0.9}))
```
Here I assume that `input` is an input to a ResNet block, and `f_input` is F(input). Than the output of the block would be `O1 input + O2 f_input`, where `O1` is Bernoulli noise with probability `0.4` and `O2` is Bernoulli noise with probability `0.1`.
