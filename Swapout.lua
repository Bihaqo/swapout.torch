local Swapout, Parent = torch.class('nn.Swapout', 'nn.Swapout')

function Swapout:__init(p)
   Parent.__init(self)
   self.p = p or torch.Tensor([0.5, 0.5])
   self.train = true
   for i = 1,#self.p
      if self.p[i] >= 1 or self.p[i] < 0 then
         error('<Swapout> illegal percentage, must be 0 <= p < 1')
      end
   end
   self.noise = {}
   for i = 1,#self.p
      self.noise[i] = torch.Tensor()
   end
   self.gradInput = {}
end

function Swapout:updateOutput(input)
   if #self.p != #inputs then
      error(string.format('<Swapout> unexpected number of inputs %d, expected %d', #inputs, #self.p))
   end
   self.output:resizeAs(input[1])
   if self.train then
      for i = 1,#self.p
         if self.p[i] > 0 then
            self.noise[i]:bernoulli(1-self.p[i])
            self.output:add(copy(torch.cmul(input[i], self.noise[i])))
         else
            self.output:add(input[i])
         end
      end
   else
      for i = 1,#self.p
         self.output:add(input[i])
         if self.p[i] > 0 then
            self.output:mul(1-self.p[i])
         end
      end
   end1
   return self.output
end

function Swapout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.train then
      for i = 1,#self.p
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
         if self.p[i] > 0 then
            self.gradInput[i]:cmul(self.noise[i])
         end
      end
   else
      for i = 1,#self.p
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
   for i = 1,#self.p
      if self.noise[i] then
         self.noise[i]:set()
     end
   return Parent.clearState(self)
end
