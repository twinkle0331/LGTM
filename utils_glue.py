import torch
import copy
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# kd loss
def cal_loss(s_logits, t_logits, temperature):
    soft_labels = F.log_softmax(t_logits / temperature, dim=-1, dtype=torch.float32)
    log_prob = F.log_softmax(s_logits / temperature, dim=-1, dtype=torch.float32)
    ori_kld_loss = -torch.exp(soft_labels) * log_prob + torch.exp(soft_labels) * soft_labels
    loss = torch.mean(torch.sum(ori_kld_loss, dim=-1))
    
    return loss

class LGTMTeacher(object):
    def __init__(self,teacher_model,student_model,alpha_kd,alpha_kd_t,optimizer_t,scheduler_t, temperature):
        self.temperature = temperature
        self.teacher_model = teacher_model
        self.student_model = student_model
        # for student
        self.alpha_kd = alpha_kd 
        # for teacher
        self.alpha_kd_t = alpha_kd_t 
        self.optimizer_t = optimizer_t
        self.scheduler_t = scheduler_t

    def cal_stu_tea_loss(self, teacher_outputs, student_outputs, flag=1):
        t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits
        loss, logits = student_outputs.loss, student_outputs.logits
        
        student_loss = None
        teacher_loss = None

        # if flag=0, calculate the student loss and teacher loss simultaneously
        if flag == 0:
            # for student
            t_soft_labels = t_logits.detach()
            s_kld_loss = cal_loss(logits, t_soft_labels, self.temperature)
            student_loss = self.alpha_kd * s_kld_loss + (1- self.alpha_kd) * loss
            
        # for teacher
        soft_labels = logits.detach()
        t_kld_loss = cal_loss(t_logits, soft_labels, self.temperature)
        teacher_loss = self.alpha_kd_t *  t_kld_loss + (1- self.alpha_kd_t) * t_loss
    
        return student_loss, teacher_loss
        
    def step(self, inputs, eval_inputs, network_optimizer): # network_optimizer: student's opt
        self.optimizer_t.zero_grad()
        self._backward_step_unrolled(inputs, eval_inputs, network_optimizer)
        self.optimizer_t.step()
        self.scheduler_t.step()

    def _backward_step_unrolled(self, inputs, eval_inputs, network_optimizer):
        # Copy a student model and update it
        unrolled_model, dalpha = self._compute_unrolled_model(inputs, network_optimizer)

        # Sample a batch of validation set
        for k, v in eval_inputs.items():
            eval_inputs[k] = v.to(unrolled_model.device)
        unrolled_model.train()
        outputs = unrolled_model(**eval_inputs)
        unrolled_loss = outputs[0]

        # unrolled_model: student
        unrolled_loss.backward()
        # dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # Calculate the Distillation Influence
        implicit_grads = self._hessian_vector_product(vector, inputs)

        eta = self.scheduler_t.get_last_lr()[0]
        # update teacher here, the gradients of teacher model contains two parts
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.teacher_model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
              
    def _compute_unrolled_model(self, input, network_optimizer):
        for k, v in input.items():
            input[k] = v.to(self.teacher_model.device)
        teacher_outputs = self.teacher_model(**input) 
        student_outputs = self.student_model(**input)
        student_loss, teacher_loss = self.cal_stu_tea_loss(teacher_outputs, student_outputs, flag=0)
       
        dtheta = torch.autograd.grad(student_loss, self.student_model.parameters())
        theta = []
        index = 0

        for group in network_optimizer.param_groups:
            for p in group["params"]:
                # if p.grad is None:
                #     continue
                # grad = p.grad.data
                
                grad = dtheta[index]
                index += 1
                if grad is None:
                    continue
                # grad = dtheta[index].data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = copy.deepcopy(network_optimizer.state[p])

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                d = p.data.addcdiv(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    d.add_(-group["lr"] * group["weight_decay"], d)
                theta.append(d)
        unrolled_model = self._construct_model_from_theta(_concat(theta).data)
      
        # calculate the grad for teacher
        dalpha = torch.autograd.grad(teacher_loss, self.teacher_model.parameters())
            
        return unrolled_model,dalpha

    def _construct_model_from_theta(self, theta):
        model_new = copy.deepcopy(self.student_model) # copy a student
        model_dict = model_new.state_dict()

        params, offset = {}, 0
        for k, v in self.student_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        # return model_new.cuda()
        return model_new

    def _hessian_vector_product(self, vector, input, r=1e-2):
        R = r / _concat(vector).norm() # episilon
        # vector is the gradients of the student's parameters on valuation set
        self.teacher_model.eval()
        self.student_model.eval()
        teacher_outputs = self.teacher_model(**input) # (loss), logits, (hidden_states), (attentions)

        for p, v in zip(self.student_model.parameters(), vector):
            p.data.add_(R, v)
        student_outputs = self.student_model(**input)
        _, loss_x = self.cal_stu_tea_loss(teacher_outputs, student_outputs)
        grads_p = torch.autograd.grad(loss_x, self.teacher_model.parameters())

        for p, v in zip(self.student_model.parameters(), vector):
            p.data.sub_(2 * R, v)
        teacher_outputs = self.teacher_model(**input) # (loss), logits, (hidden_states), (attentions)
        student_outputs = self.student_model(**input)
        _, loss_y = self.cal_stu_tea_loss(teacher_outputs, student_outputs)
        grads_n = torch.autograd.grad(loss_y, self.teacher_model.parameters())

        # recover the parameters of the student
        for p, v in zip(self.student_model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
