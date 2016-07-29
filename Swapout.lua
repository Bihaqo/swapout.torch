require 'nn'

local Swapout, Parent = torch.class('nn.Swapout', 'nn.Module')

function Swapout:__init(p)
   Parent.__init(self)
   self.p = p or {0.5, 0.5}
   self.train = true
   for i = 1,#self.p do
      if self.p[i] >= 1 or self.p[i] < 0 then
         error('<Swapout> illegal percentage, must be 0 <= p < 1')
      end
   end
   self.noise = {}
   for i = 1,#self.p do
      -- TODO: move to updateOutput and use input.new()?
      self.noise[i] = torch.Tensor()
   end
   self.gradInput = {}
end

function Swapout:updateOutput(input)
   if #self.p ~= #input then
      error(string.format('<Swapout> unexpected number of inputs %d, expected %d', #inputs, #self.p))
   end
  --  TODO: (speedup) copy() the first input[1] instead of zero() and add().
   self.output:resizeAs(input[1]):zero()
   if self.train then
      for i = 1,#self.p do
         if self.p[i] > 0 then
            self.noise[i]:resizeAs(input[i])
            self.noise[i]:bernoulli(1-self.p[i])
            self.output:add(torch.cmul(input[i], self.noise[i]))
         else
            self.output:add(input[i])
         end
      end
   else
      for i = 1,#self.p do
         if self.p[i] > 0 then
            self.output:add(1-self.p[i], input[i])
         else
            self.output:add(input[i])
         end
      end
   end
   return self.output
end

function Swapout:updateGradInput(input, gradOutput)
   if self.train then
      for i = 1,#self.p do
         self.gradInput[i] = input[i].new()
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
         if self.p[i] > 0 then
            self.gradInput[i]:cmul(self.noise[i])
         end
      end
   else
      for i = 1,#self.p do
         self.gradInput[i] = input[i].new()
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
         if self.p[i] > 0 then
            self.gradInput[i]:mul(1-self.p[i])
         end
       end
   end
   return self.gradInput
end

function Swapout:setp(p)
   self.p = p
end

function Swapout:__tostring__()
-- TODO: output vector of probabilities.
   return string.format('%s', torch.type(self))--, self.p1, self.p2)
end


function Swapout:clearState()
   for i = 1,#self.p do
      if self.noise[i] then
         self.noise[i]:set()
     end
   end
  --  TODO: clean self.gradInput?
   return Parent.clearState(self)
end
