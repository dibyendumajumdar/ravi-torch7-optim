require 'torch'
require 'optim'
local checks = 0
local originalDefault = torch.getdefaulttensortype()
for _,typename in ipairs{'torch.DoubleTensor','torch.FloatTensor'} do
  torch.setdefaulttensortype(typename == 'torch.FloatTensor' and 'torch.DoubleTensor' or 'torch.FloatTensor')
  for _,accelerated in ipairs{false,true} do
    for _,lambda in ipairs{0,0.5} do
      local target = torch.Tensor{-3,0.25,2}:type(typename)
      local expected = target:clone()
      local function shrink(x, threshold)
        for i=1,x:nElement() do
          local v=x[i]
          x[i]=(v<0 and -1 or 1)*math.max(math.abs(v)-threshold,0)
        end
      end
      shrink(expected,lambda)
      local function f(x)
        local delta=x-target
        return 0.5*delta:dot(delta),delta
      end
      local function g(x) return lambda*x:clone():abs():sum() end
      local function prox(x,L) shrink(x,lambda/L) end
      local x=torch.zeros(3):type(typename)
      local result,history=optim.FistaLS(f,g,prox,x,{L=2,maxiter=200,errthres=1e-12,doFistaUpdate=accelerated})
      local err=(result-expected):abs():max()
      -- Float objective values can stop changing before coordinates fully converge.
      local tolerance = typename == 'torch.FloatTensor' and 5e-4 or 1e-5
      assert(err<tolerance,'FISTA solution error: '..err)
      assert(#history>0 and history[#history].F==history[#history].F)
      checks=checks+1
    end
  end
end
torch.setdefaulttensortype(originalDefault)
print('FISTA/ISTA known-solution cases passed: '..checks)
